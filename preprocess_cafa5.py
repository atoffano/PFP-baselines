#!/usr/bin/env python3
"""
preprocessing script for CAFA5 data integration.

Performs:
1. Extraction & staging of CAFA5 full_evaluation assets (Zenodo data).
2. Creation of dataset directories:
   - data/CAFA5-NK/
   - data/CAFA5-LK/
   - data/CAFA5-PK/
3. Generation of test annotation files (.tsv and .pkl) for BPO, CCO, MFO.
4. Generation of standalone training annotations (.tsv) from Train/train_terms.tsv,
   filtering out category-specific test proteins to prevent leakage.
5. Generation of background_<dataset>.pkl files using background.py.
6. Alignment of missing test target proteins against SwissProt 2024_01 via DIAMOND
   and injection of supplementary hits into diamond_swissprot_2024_01_alignment.tsv (some cafa sequences come from trembl).
"""

import os
import sys
import pickle
import tarfile
import shutil
import subprocess
from collections import defaultdict
import pandas as pd
from Bio import SeqIO

DATA_DIR = "./data"
CAFA5_DIR = os.path.join(DATA_DIR, "CAFA5")
FULL_EVAL_DIR = os.path.join(CAFA5_DIR, "full_evaluation")
TRAIN_TERMS_PATH = os.path.join(CAFA5_DIR, "Train", "train_terms.tsv")
TEST_FASTA_PATH = os.path.join(CAFA5_DIR, "Test (Targets)", "testsuperset.fasta")
SWISSPROT_DIR = os.path.join(DATA_DIR, "swissprot", "2024_01")
SWISSPROT_DMND = os.path.join(SWISSPROT_DIR, "swissprot_2024_01.dmnd")
ALIGNMENT_TSV = os.path.join(SWISSPROT_DIR, "diamond_swissprot_2024_01_alignment.tsv")

SETTINGS = {
    "CAFA5-NK": "eval_terms_no_knowledge_2025_03.tsv",
    "CAFA5-LK": "eval_terms_limited_2025_03.tsv",
    "CAFA5-PK": "eval_terms_partial_2025_03.tsv",
}

ASPECTS = ["BPO", "CCO", "MFO"]
ONTOLOGY_MAP = {"BPO": "all_bp", "CCO": "all_cc", "MFO": "all_mf"}


def stage_evaluation_assets():
    """Unpack Zenodo full_evaluation archive if not already unpacked."""
    print("=== Step 1: Checking and staging CAFA5 evaluation assets ===")
    os.makedirs(CAFA5_DIR, exist_ok=True)
    os.makedirs(FULL_EVAL_DIR, exist_ok=True)

    archive_candidates = [
        os.path.join(CAFA5_DIR, "full_evaluation.tar.gz"),
        "/tmp/full_evaluation.tar.gz",
    ]
    
    needed_files = [
        "go-basic.obo", "IA_t0.tsv", "known_t0.tsv", "toi_2025_03.tsv",
        "eval_terms_no_knowledge_2025_03.tsv",
        "eval_terms_limited_2025_03.tsv",
        "eval_terms_partial_2025_03.tsv",
    ]

    missing = [f for f in needed_files if not os.path.exists(os.path.join(FULL_EVAL_DIR, f))]
    if missing:
        tar_path = None
        for cand in archive_candidates:
            if os.path.exists(cand):
                tar_path = cand
                break

        if not tar_path:
            url = "https://zenodo.org/records/20186533/files/full_evaluation.tar.gz?download=1"
            tar_path = os.path.join(CAFA5_DIR, "full_evaluation.tar.gz")
            print(f"Downloading Zenodo full_evaluation benchmark archive from {url}...")
            subprocess.run(["curl", "-sL", url, "-o", tar_path], check=True)

        print(f"Extracting {tar_path} into {CAFA5_DIR}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=CAFA5_DIR)

    # Copy shared evaluation metadata into CAFA5 root for convenience
    for fname in ["go-basic.obo", "IA_t0.tsv", "known_t0.tsv", "toi_2025_03.tsv"]:
        src = os.path.join(FULL_EVAL_DIR, fname)
        dst = os.path.join(CAFA5_DIR, fname)
        if not os.path.exists(dst) and os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied {fname} to {CAFA5_DIR}")


def create_test_files():
    """Generate test TSV and PKL files for CAFA5-NK, CAFA5-LK, CAFA5-PK."""
    print("\n=== Step 2: Creating Test Annotation Files ===")
    test_proteins_by_setting = {}

    for dataset_name, eval_file_name in SETTINGS.items():
        dataset_dir = os.path.join(DATA_DIR, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        eval_path = os.path.join(FULL_EVAL_DIR, eval_file_name)
        print(f"Loading {eval_path} for {dataset_name}...")
        df_eval = pd.read_csv(eval_path, sep="\t")

        test_proteins_by_setting[dataset_name] = set(df_eval["EntryID"].unique())
        print(f"  {dataset_name}: {len(test_proteins_by_setting[dataset_name])} unique test proteins")

        for aspect in ASPECTS:
            df_aspect = df_eval[df_eval["aspect"] == aspect]
            tsv_out = os.path.join(dataset_dir, f"{dataset_name}_{aspect}_test_annotations.tsv")
            pkl_out = os.path.join(dataset_dir, f"{dataset_name}_{aspect}_test_annotations.pkl")

            grouped = df_aspect.groupby("EntryID")["term"].apply(
                lambda s: "; ".join(sorted(set(s)))
            ).reset_index()

            grouped.to_csv(tsv_out, sep="\t", index=False)
            print(f"    Saved {tsv_out} ({len(grouped)} proteins)")

            ont_key = ONTOLOGY_MAP[aspect]
            protein_go_terms = defaultdict(lambda: {"all_bp": set(), "all_cc": set(), "all_mf": set()})
            for _, row in grouped.iterrows():
                pid = row["EntryID"]
                terms = set(row["term"].split("; "))
                protein_go_terms[pid][ont_key] = terms

            with open(pkl_out, "wb") as f:
                pickle.dump(dict(protein_go_terms), f)
            print(f"    Saved {pkl_out}")

    return test_proteins_by_setting


def create_train_files(test_proteins_by_setting):
    """Generate training TSV files and background pickle for each setting."""
    print("\n=== Step 3: Creating Training Annotations and Background Files ===")
    print(f"Loading {TRAIN_TERMS_PATH}...")
    df_train = pd.read_csv(TRAIN_TERMS_PATH, sep="\t")
    print(f"Total training annotations: {len(df_train)} ({df_train['EntryID'].nunique()} unique proteins)")

    print("Aggregating training terms by (EntryID, aspect)...")
    grouped_train = df_train.groupby(["EntryID", "aspect"])["term"].apply(
        lambda s: "; ".join(sorted(set(s)))
    ).reset_index()

    for dataset_name, test_proteins in test_proteins_by_setting.items():
        dataset_dir = os.path.join(DATA_DIR, dataset_name)
        print(f"\nProcessing training set for {dataset_name} (excluding {len(test_proteins)} test proteins)...")

        df_filtered = grouped_train[~grouped_train["EntryID"].isin(test_proteins)]
        print(f"  {dataset_name} remaining train proteins: {df_filtered['EntryID'].nunique()}")

        for aspect in ASPECTS:
            df_aspect = df_filtered[df_filtered["aspect"] == aspect][["EntryID", "term"]]
            tsv_out = os.path.join(dataset_dir, f"{dataset_name}_{aspect}_train_annotations.tsv")
            df_aspect.to_csv(tsv_out, sep="\t", index=False)
            print(f"    Saved {tsv_out} ({len(df_aspect)} proteins)")

        # Create background_<dataset>.pkl
        bg_out = os.path.join(dataset_dir, f"background_{dataset_name}.pkl")
        bpo_train = os.path.join(dataset_dir, f"{dataset_name}_BPO_train_annotations.tsv")
        cco_train = os.path.join(dataset_dir, f"{dataset_name}_CCO_train_annotations.tsv")
        mfo_train = os.path.join(dataset_dir, f"{dataset_name}_MFO_train_annotations.tsv")
        bpo_test = os.path.join(dataset_dir, f"{dataset_name}_BPO_test_annotations.tsv")
        cco_test = os.path.join(dataset_dir, f"{dataset_name}_CCO_test_annotations.tsv")
        mfo_test = os.path.join(dataset_dir, f"{dataset_name}_MFO_test_annotations.tsv")

        cmd = [
            sys.executable,
            "background.py",
            "--cco", cco_train,
            "--bpo", bpo_train,
            "--mfo", mfo_train,
            "--test_cco", cco_test,
            "--test_bpo", bpo_test,
            "--test_mfo", mfo_test,
            "--output", bg_out,
        ]
        print(f"    Running background.py for {dataset_name}...")
        subprocess.run(cmd, check=True)
        print(f"    Saved {bg_out}")


def align_and_inject_missing_diamond(test_proteins_by_setting):
    """Align missing test target sequences against SwissProt 2024_01 and append to alignment TSV."""
    print("\n=== Step 4: DIAMOND Alignment ===")
    all_eval_proteins = set().union(*test_proteins_by_setting.values())
    print(f"Total unique evaluation proteins across all CAFA5 settings: {len(all_eval_proteins)}")

    if not os.path.exists(ALIGNMENT_TSV):
        print(f"Warning: Alignment file {ALIGNMENT_TSV} not found. Skipping DIAMOND injection.")
        return

    print(f"Scanning {ALIGNMENT_TSV} for existing query proteins...")
    existing_queries = set()
    with open(ALIGNMENT_TSV, "r") as f:
        for line in f:
            q = line.split("\t", 1)[0]
            if q in all_eval_proteins:
                existing_queries.add(q)
                if len(existing_queries) == len(all_eval_proteins):
                    break

    missing_pids = all_eval_proteins - existing_queries
    print(f"  Proteins already aligned: {len(existing_queries)} / {len(all_eval_proteins)} ({len(existing_queries)/len(all_eval_proteins)*100:.2f}%)")
    print(f"  Missing target proteins: {len(missing_pids)}")

    if not missing_pids:
        print("  All evaluation proteins already have DIAMOND alignment hits. Nothing to inject.")
        return

    if not os.path.exists(TEST_FASTA_PATH):
        print(f"Warning: {TEST_FASTA_PATH} not found. Cannot extract missing sequences.")
        return

    missing_fasta = os.path.join(CAFA5_DIR, "missing_eval_targets.fasta")
    missing_aln_tsv = os.path.join(CAFA5_DIR, "missing_eval_targets_alignment.tsv")

    print(f"Extracting {len(missing_pids)} sequences from {TEST_FASTA_PATH}...")
    records_to_align = []
    with open(TEST_FASTA_PATH, "r") as f:
        for rec in SeqIO.parse(f, "fasta"):
            pid = rec.id.split()[0]
            if pid in missing_pids:
                rec.id = pid
                rec.description = ""
                records_to_align.append(rec)

    SeqIO.write(records_to_align, missing_fasta, "fasta")
    print(f"  Saved {len(records_to_align)} records to {missing_fasta}")

    if not os.path.exists(SWISSPROT_DMND):
        print(f"Warning: DIAMOND database {SWISSPROT_DMND} not found. Cannot run alignment.")
        return

    diamond_cmd = [
        "diamond", "blastp",
        "-d", SWISSPROT_DMND,
        "-q", missing_fasta,
        "-o", missing_aln_tsv,
        "-e", "0.001",
        "--outfmt", "6", "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen", "qstart", "qend", "sstart", "send", "evalue", "bitscore",
        "--threads", "8",
    ]
    print(f"Running DIAMOND blastp on missing sequences...")
    subprocess.run(diamond_cmd, check=True)

    # Append alignments
    print(f"Appending new alignments from {missing_aln_tsv} to {ALIGNMENT_TSV}...")
    with open(missing_aln_tsv, "r") as src, open(ALIGNMENT_TSV, "a") as dst:
        shutil.copyfileobj(src, dst)
    print("  DIAMOND supplementary alignments injected successfully.")


def main():
    stage_evaluation_assets()
    test_proteins_by_setting = create_test_files()
    create_train_files(test_proteins_by_setting)
    align_and_inject_missing_diamond(test_proteins_by_setting)
    print("\n===========================================================")
    print("✓ CAFA5 Preprocessing and Alignment Injection Complete!")
    print("===========================================================")


if __name__ == "__main__":
    main()
