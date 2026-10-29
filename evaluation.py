import pandas as pd
import os
import sys
import subprocess
import pickle
from collections import defaultdict
import tqdm
import argparse
import logging


def setup_logging(output_dir, aspect):
    """Setup logging for evaluation."""
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"{aspect}")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    log_file = os.path.join(log_dir, f"{aspect}_eval.log")
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def evaluate(
    logger, output_dir, dataset, aspect, k_values, eval_type="beprof", norm="cafa"
):
    """
    Evaluate predictions with the selected evaluator.
    """
    go_obo_file = "./data/go.obo"
    if dataset.startswith("CAFA5") and os.path.exists("./data/CAFA5/go-basic.obo"):
        go_obo_file = "./data/CAFA5/go-basic.obo"

    known_file = None
    if dataset == "CAFA5-PK":
        if os.path.exists("./data/CAFA5/known_t0.tsv"):
            known_file = "./data/CAFA5/known_t0.tsv"
        elif os.path.exists("./data/CAFA5/full_evaluation/known_t0.tsv"):
            known_file = "./data/CAFA5/full_evaluation/known_t0.tsv"

    ia_file = None
    if dataset.startswith("CAFA5"):
        if os.path.exists("./data/CAFA5/IA_t0.tsv"):
            ia_file = "./data/CAFA5/IA_t0.tsv"
        elif os.path.exists("./data/CAFA5/full_evaluation/IA_t0.tsv"):
            ia_file = "./data/CAFA5/full_evaluation/IA_t0.tsv"

    toi_file = None
    if dataset.startswith("CAFA5"):
        if os.path.exists("./data/CAFA5/full_evaluation/toi_2025_03.tsv"):
            toi_file = "./data/CAFA5/full_evaluation/toi_2025_03.tsv"
        elif os.path.exists("./data/CAFA5/toi_2025_03.tsv"):
            toi_file = "./data/CAFA5/toi_2025_03.tsv"

    gt_tsv = f"./data/{dataset}/{dataset}_{aspect}_test_annotations.tsv"

    if not os.path.exists(gt_tsv):
        logger.error(f"Ground Truth TSV file {gt_tsv} does not exist.")
        raise FileNotFoundError(f"Ground Truth TSV file {gt_tsv} does not exist.")

    cafa5_gt_map = {
        "CAFA5-NK": "./data/CAFA5/full_evaluation/eval_terms_no_knowledge_2025_03.tsv",
        "CAFA5-LK": "./data/CAFA5/full_evaluation/eval_terms_limited_2025_03.tsv",
        "CAFA5-PK": "./data/CAFA5/full_evaluation/eval_terms_partial_2025_03.tsv",
    }
    cafa_gt_tsv = gt_tsv
    if dataset.startswith("CAFA5") and dataset in cafa5_gt_map and os.path.exists(cafa5_gt_map[dataset]):
        cafa_gt_tsv = cafa5_gt_map[dataset]

    background_pkl = f"./data/{dataset}/background_{dataset}.pkl"
    gt_pkl = f"./data/{dataset}/{dataset}_{aspect}_test_annotations.pkl"
    all_test_proteins = None

    if eval_type == "beprof":
        # Check if GT exists in pkl format. If not, convert GT TSV to pkl using gt_convert
        if not os.path.exists(gt_pkl):
            logger.info(f"Converting Ground Truth TSV {gt_tsv} to pkl format")
            gt_convert(gt_tsv)

        # Load GT protein IDs so we can fill in empty entries for unaligned proteins
        with open(gt_pkl, "rb") as f:
            gt_data = pickle.load(f)
        all_test_proteins = set(gt_data.keys())
        logger.info(
            f"{len(all_test_proteins)} test proteins loaded from ground truth pkl"
        )

    logger.info(f"Using evaluator: {eval_type}")

    # Evaluate NaiveBaseline predictions
    logger.info(f"Evaluating NaiveBaseline predictions")
    pred_file = f"{output_dir}/predictions/NaiveBaseline/predictions.tsv"
    pred_pkl = f"{output_dir}/predictions/NaiveBaseline/predictions.pkl"
    if os.path.exists(pred_file):
        if eval_type == "beprof":
            pred_dict = convert_predictions(pred_file, aspect, all_test_proteins)
            with open(pred_pkl, "wb") as f:
                pickle.dump(pred_dict, f)

            run_beprof_evaluation(
                logger,
                pred_pkl,
                gt_pkl,
                background_pkl,
                go_obo_file,
                f"{output_dir}/evaluation/NaiveBaseline",
            )
        else:
            run_cafa_evaluation(
                logger,
                pred_file,
                cafa_gt_tsv,
                go_obo_file,
                f"{output_dir}/evaluation/NaiveBaseline",
                norm=norm,
                known_file=known_file,
                ia_file=ia_file,
                toi_file=toi_file,
                dataset=dataset,
            )
    else:
        logger.warning(f"NaiveBaseline predictions file {pred_file} does not exist.")

    # Evaluate IDScore predictions
    logger.info(f"Evaluating IDScore predictions")
    # Convert predictions to pkl
    pred_file = f"{output_dir}/predictions/IDScore/predictions.tsv"
    pred_pkl = f"{output_dir}/predictions/IDScore/predictions.pkl"
    if os.path.exists(pred_file):
        if eval_type == "beprof":
            pred_dict = convert_predictions(pred_file, aspect, all_test_proteins)
            with open(pred_pkl, "wb") as f:
                pickle.dump(pred_dict, f)

            run_beprof_evaluation(
                logger,
                pred_pkl,
                gt_pkl,
                background_pkl,
                go_obo_file,
                f"{output_dir}/evaluation/IDScore",
            )
        else:
            run_cafa_evaluation(
                logger,
                pred_file,
                cafa_gt_tsv,
                go_obo_file,
                f"{output_dir}/evaluation/IDScore",
                norm=norm,
                known_file=known_file,
                ia_file=ia_file,
                toi_file=toi_file,
                dataset=dataset,
            )
    else:
        logger.warning(f"IDScore predictions file {pred_file} does not exist.")

    # Evaluate AlignmentScore predictions
    logger.info(f"Evaluating AlignmentScore predictions")
    pred_file = f"{output_dir}/predictions/AlignmentScore/predictions.tsv"
    pred_pkl = f"{output_dir}/predictions/AlignmentScore/predictions.pkl"
    if os.path.exists(pred_file):
        if eval_type == "beprof":
            pred_dict = convert_predictions(pred_file, aspect, all_test_proteins)
            with open(pred_pkl, "wb") as f:
                pickle.dump(pred_dict, f)
            run_beprof_evaluation(
                logger,
                pred_pkl,
                gt_pkl,
                background_pkl,
                go_obo_file,
                f"{output_dir}/evaluation/AlignmentScore",
            )
        else:
            run_cafa_evaluation(
                logger,
                pred_file,
                cafa_gt_tsv,
                go_obo_file,
                f"{output_dir}/evaluation/AlignmentScore",
                norm=norm,
                known_file=known_file,
                ia_file=ia_file,
                toi_file=toi_file,
                dataset=dataset,
            )
    else:
        logger.warning(f"AlignmentScore predictions file {pred_file} does not exist.")

    # Evaluate BlastKNN predictions for each k value
    for k in k_values:
        logger.info(f"Evaluating BlastKNN predictions for k={k}")
        pred_file = f"{output_dir}/predictions/BlastKNN/k{k}_predictions.tsv"
        pred_pkl = f"{output_dir}/predictions/BlastKNN/k{k}_predictions.pkl"
        if os.path.exists(pred_file):
            if eval_type == "beprof":
                pred_dict = convert_predictions(pred_file, aspect, all_test_proteins)
                with open(pred_pkl, "wb") as f:
                    pickle.dump(pred_dict, f)

                run_beprof_evaluation(
                    logger,
                    pred_pkl,
                    gt_pkl,
                    background_pkl,
                    go_obo_file,
                    f"{output_dir}/evaluation/BlastKNN_k{k}",
                )
            else:
                run_cafa_evaluation(
                    logger,
                    pred_file,
                    cafa_gt_tsv,
                    go_obo_file,
                    f"{output_dir}/evaluation/BlastKNN_k{k}",
                    norm=norm,
                    known_file=known_file,
                    ia_file=ia_file,
                    toi_file=toi_file,
                    dataset=dataset,
                )
        else:
            logger.warning(
                f"BlastKNN predictions file {pred_file} for k={k} does not exist."
            )


def run_cafa_evaluation(
    logger,
    pred_tsv,
    gt_tsv,
    go_obo_file,
    eval_output_dir,
    norm="cafa",
    known_file=None,
    ia_file=None,
    toi_file=None,
    dataset=None,
):
    """
    Run CAFA evaluation as a subprocess.
    If evaluating CAFA5 (dataset starts with 'CAFA5'), use claradepaolis/CAFA-evaluator-PK.
    Otherwise, fall back to in-tree cafa_eval.py.
    """
    os.makedirs(eval_output_dir, exist_ok=True)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cafa_pk_dir = os.path.join(base_dir, "CAFA-evaluator-PK")
    is_cafa5 = dataset is not None and dataset.startswith("CAFA5")

    if is_cafa5 and os.path.exists(os.path.join(cafa_pk_dir, "src", "cafaeval", "__main__.py")):
        import tempfile
        import shutil

        # CAFA-evaluator-PK expects a folder containing the prediction file(s)
        # We stage the single prediction file in a clean temp directory
        with tempfile.TemporaryDirectory(prefix="cafa5_eval_") as tmp_pred_dir:
            tmp_pred_file = os.path.join(tmp_pred_dir, "predictions.tsv")
            try:
                os.symlink(os.path.abspath(pred_tsv), tmp_pred_file)
            except OSError:
                shutil.copyfile(pred_tsv, tmp_pred_file)

            env = os.environ.copy()
            cafa_pk_src = os.path.join(cafa_pk_dir, "src")
            env["PYTHONPATH"] = f"{cafa_pk_src}:{env.get('PYTHONPATH', '')}"

            eval_threads = "2" if dataset == "CAFA5-PK" else "4"
            cmd = [
                sys.executable,
                "-m",
                "cafaeval",
                go_obo_file,
                tmp_pred_dir,
                gt_tsv,
                "-out_dir",
                eval_output_dir,
                "-norm",
                norm,
                "-threads",
                eval_threads,
            ]
            if ia_file and os.path.exists(ia_file):
                cmd.extend(["-ia", ia_file])
            if toi_file and os.path.exists(toi_file):
                cmd.extend(["-toi", toi_file])
            if known_file and os.path.exists(known_file):
                cmd.extend(["-known", known_file])

            logger.info(f"Running CAFA-evaluator-PK evaluation: {' '.join(cmd)}")
            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    env=env,
                    cwd=base_dir,
                )
                logger.info("CAFA-evaluator-PK evaluation completed successfully")
                logger.info(f"Results saved to: {eval_output_dir}")
                if result.stdout:
                    logger.info(f"CAFA-evaluator-PK stdout: {result.stdout}")
            except subprocess.CalledProcessError as e:
                logger.error(
                    f"CAFA-evaluator-PK evaluation failed with return code {e.returncode}"
                )
                if e.stdout:
                    logger.error(f"CAFA-evaluator-PK stdout:\n{e.stdout}")
                if e.stderr:
                    logger.error(f"CAFA-evaluator-PK stderr:\n{e.stderr}")
                raise
            except Exception as e:
                logger.error(f"Error running CAFA-evaluator-PK evaluation: {str(e)}")
                raise
    else:
        cmd = [
            sys.executable,
            "cafa_eval.py",
            "--predictions",
            pred_tsv,
            "--ground_truth",
            gt_tsv,
            "--ontology",
            go_obo_file,
            "--output",
            eval_output_dir,
            "--norm",
            norm,
        ]
        if known_file and os.path.exists(known_file):
            cmd.extend(["--known", known_file])
        if ia_file and os.path.exists(ia_file):
            cmd.extend(["--ia", ia_file])

        logger.info(f"Running CAFA evaluation: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=base_dir,
            )
            logger.info("CAFA evaluation completed successfully")
            logger.info(f"Results saved to: {eval_output_dir}")
            if result.stdout:
                logger.info(f"CAFA stdout: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"CAFA evaluation failed with return code {e.returncode}")
            if e.stdout:
                logger.error(f"CAFA stdout:\n{e.stdout}")
            if e.stderr:
                logger.error(f"CAFA stderr:\n{e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Error running CAFA evaluation: {str(e)}")
            raise


def run_beprof_evaluation(
    logger, pred_pkl, gt_pkl, background_pkl, go_obo_file, eval_output_dir
):
    """
    Run beprof_eval.py as a subprocess.
    """
    os.makedirs(eval_output_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "beprof_eval.py",
        "--predict",
        pred_pkl,
        "--output_path",
        eval_output_dir,
        "--true",
        gt_pkl,
        "--background",
        background_pkl,
        "--go",
        go_obo_file,
        "--metrics",
        "0,1,2,3,4,5",  # All metrics: F_max, Smin, Aupr, ICAupr, DPAupr, threshold
    ]

    logger.info(f"Running BeProf evaluation: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),  # Run from script directory
        )

        logger.info(f"BeProf evaluation completed successfully")
        logger.info(f"Results saved to: {eval_output_dir}")

        # Log stdout if there's any output
        if result.stdout:
            logger.info(f"BeProf stdout: {result.stdout}")

    except subprocess.CalledProcessError as e:
        logger.error(f"BeProf evaluation failed with return code {e.returncode}")
        if e.stdout:
            logger.error(f"BeProf stdout:\n{e.stdout}")
        if e.stderr:
            logger.error(f"BeProf stderr:\n{e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Error running BeProf evaluation: {str(e)}")
        raise


def gt_convert(gt_tsv):
    """
    Convert GT to pkl.
    """
    base_name = os.path.basename(gt_tsv)
    ontology_type = base_name.split("_")[1]  # Extract BPO, CCO, or MFO
    ontology_mapping = {"BPO": "all_bp", "CCO": "all_cc", "MFO": "all_mf"}
    ontology_key = ontology_mapping[ontology_type]

    gt_pkl = os.path.splitext(gt_tsv)[0] + ".pkl"

    print(f"Processing {base_name}...")

    # Read the input file
    df = pd.read_csv(gt_tsv, sep="\t")
    df["term"] = df["term"].str.split("; ")
    df = df.explode("term")

    protein_go_terms = defaultdict(
        lambda: {"all_bp": set(), "all_cc": set(), "all_mf": set()}
    )
    for _, row in df.iterrows():
        protein_id = row["EntryID"]
        term_id = row["term"]
        protein_go_terms[protein_id][ontology_key].add(term_id)

    protein_go_terms_dict = dict(protein_go_terms)

    # Save to pickle file
    with open(gt_pkl, "wb") as f:
        pickle.dump(protein_go_terms_dict, f)
    print(f"Saved pickle file: {gt_pkl}")


def convert_predictions(pred_file, aspect, all_test_proteins=None):
    """
    Converts a TSV prediction file with columns: target_ID, term_ID, score
    into a dictionary where each protein maps to a {subontology: {term: score}} dict.

    Proteins that received no predictions are included with an empty prediction dict
    so that beprof_eval penalises them via recall rather than silently ignoring them.

    Parameters
    ----------
    pred_file : str
        Path to the TSV prediction file.
    aspect : str
        Ontology aspect string, e.g. 'BPO', 'CCO', 'MFO'.
    all_test_proteins : set or None
        Complete set of test protein IDs. When provided, every protein absent from
        pred_file is inserted with an empty prediction dict.
    """
    subontology = aspect[:2].lower()
    df = pd.read_csv(pred_file, sep="\t")
    # Split term column by '; ' and explode
    df["term_ID"] = df["term_ID"].str.split("; ")
    df = df.explode("term_ID")
    pred_dict = {}
    for prot, group in tqdm.tqdm(df.groupby("target_ID")):
        pred_dict[prot] = {
            f"{subontology}": dict(zip(group["term_ID"], group["score"]))
        }

    # Ensure every test protein has an entry so beprof_eval counts absences as
    # zero-precision / zero-recall rather than excluding them from the evaluation.
    if all_test_proteins is not None:
        n_missing = 0
        for prot in all_test_proteins:
            if prot not in pred_dict:
                pred_dict[prot] = {f"{subontology}": {}}
                n_missing += 1
        if n_missing:
            print(
                f"Added {n_missing} proteins with no predictions "
                f"as empty entries (aspect={aspect})."
            )
    return pred_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictions.")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing predictions and evaluation results. Will also serve as output",
    )
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument(
        "--aspect", required=True, help="Ontology aspect (BPO, CCO, MFO)."
    )
    parser.add_argument(
        "--k_values",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10, 15, 20],
        help="List of k values for BlastKNN.",
    )
    parser.add_argument(
        "--eval",
        choices=["beprof", "cafa"],
        default="beprof",
        help="Evaluation backend to use.",
    )
    parser.add_argument(
        "--norm",
        choices=["cafa", "pred", "gt"],
        default="cafa",
        help="Normalization mode for CAFA evaluation.",
    )
    args = parser.parse_args()

    logger = setup_logging(args.input_dir, args.aspect)
    evaluate(
        logger,
        args.input_dir,
        args.dataset,
        args.aspect,
        args.k_values,
        eval_type=args.eval,
        norm=args.norm,
    )
