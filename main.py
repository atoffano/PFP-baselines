import pandas as pd
import os
import tqdm
import argparse
import logging
from constants import *
from dataloading import (
    load_data,
    load_pairwise_alignment,
    load_uniprot_mapping,
    load_stringdb,
)
import methods
import evaluation


def setup_logging(output_dir, aspect):
    """Setup logging for each aspect with separate log files."""
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"{aspect}")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    log_file = os.path.join(log_dir, f"{aspect}.log")
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


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline annotation transfer methods.",
        allow_abbrev=False,
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[],
        help="List of k closest sequences to transfer annotations from.",
    )
    parser.add_argument(
        "--db_versions",
        type=str,
        nargs="+",
        default=SWISSPROT_VERSIONS,
        help="SwissProt DB versions to process.",
    )
    parser.add_argument(
        "--aspects",
        type=str,
        nargs="+",
        default=["BPO", "CCO", "MFO"],
        help="Ontology aspects to process.",
    )
    parser.add_argument(
        "--one_vs_all",
        action="store_true",
        help="Whether to use one-vs-all approach for annotation propagation.",
    )
    parser.add_argument(
        "--annotations_2024_01",
        action="store_true",
        help="Whether to fix train proteins' annotations to 2024 SwissProt version.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Appended suffix to the output directory.",
    )
    parser.add_argument(
        "--experimental_only",
        action="store_true",
        help="Whether to use only experimental annotations.",
    )

    parser.add_argument(
        "--stringdb_mode",
        type=str,
        default=None,
        choices=["rescue", "merge"],
        help=(
            "Incorporate STRING DB combined_score into annotation transfer. "
            "'rescue': use StringDB only for proteins with no alignment hit. "
            "'merge': blend alignment and StringDB scores (set weight with --stringdb_weight). "
            "Examples: --stringdb_mode rescue  |  --stringdb_mode merge --stringdb_weight 0.5"
        ),
    )
    parser.add_argument(
        "--stringdb_weight",
        type=float,
        default=0.5,
        help="Weight for StringDB scores when --stringdb_mode merge is used (default: 0.5, must be in [0, 1]).",
    )

    args = parser.parse_args()

    # Validate --stringdb arguments
    stringdb_mode = args.stringdb_mode
    stringdb_weight = args.stringdb_weight
    if stringdb_mode == "merge" and not 0.0 <= stringdb_weight <= 1.0:
        raise ValueError(f"--stringdb_weight must be in [0, 1], got {stringdb_weight}")

    # Mapping from SwissProt Entry Name (e.g. Q6GZX1) to EntryID (004R_FRG3G)
    id_mapping = load_uniprot_mapping()

    for db_version in tqdm.tqdm(args.db_versions, desc="Processing databases"):
        for aspect in args.aspects:

            output_dir = f"./results/{args.dataset}/baselines_{args.dataset}_{db_version}_{aspect}{args.output_suffix}"
            if args.experimental_only:
                output_dir += "_exp"
            if args.annotations_2024_01:
                output_dir += "_2024_annotations"
            if args.one_vs_all:
                output_dir += "_one_vs_all"
            if stringdb_mode == "rescue":
                output_dir += "_stringdb_rescue"
            elif stringdb_mode == "merge":
                output_dir += f"_stringdb_merge_{stringdb_weight}"
            os.makedirs(f"{output_dir}/predictions", exist_ok=True)

            # Setup logging for this aspect
            logger = setup_logging(output_dir, aspect)
            logger.info(
                f"Starting processing for {aspect} with database version {db_version}"
            )

            # Load data
            logger.info(f"Loading data for {args.dataset} with aspect {aspect}")
            train, test = load_data(
                logger,
                args.dataset,
                aspect,
                db_version,
                annotations_2024_01=args.annotations_2024_01,
                id_mapping=id_mapping,
                experimental_only=args.experimental_only,
                one_vs_all=args.one_vs_all,
            )
            logger.info(
                f"Loaded {train['EntryID'].nunique()} training proteins and {test['EntryID'].nunique()} test proteins"
            )
            logger.info(f"One-vs-All approach: {args.one_vs_all}.")
            logger.info(f"Experimental annotations only: {args.experimental_only}")
            logger.info(f"SwissProt 2024 annotations: {args.annotations_2024_01}")
            logger.info(f"Output directory: {output_dir}")
            logger.info(f"Train set:\n{train}")
            logger.info(f"Test set:\n{test}")

            # Comment out to skip Naive Baseline
            # logger.info("Running Naive Baseline...")
            # os.makedirs(f"{output_dir}/predictions/NaiveBaseline", exist_ok=True)
            # methods.naive_baseline(output_dir, train, test)
            # logger.info(f"Naive Baseline predictions saved to {output_dir}/predictions")
            # ---

            logger.info("Loading pairwise alignments...")
            pairwise_alignment = load_pairwise_alignment(
                args.dataset,
                id_mapping=id_mapping,
            )

            # Load StringDB data once per aspect (independent of db_version)
            stringdb_data = None
            if stringdb_mode is not None:
                logger.info(f"Loading StringDB interactions (mode={stringdb_mode})...")
                stringdb_data = load_stringdb(args.dataset, id_mapping=id_mapping)
                stringdb_data = stringdb_data[
                    stringdb_data["query_id"].isin(test["EntryID"].unique())
                    & stringdb_data["subject_id"].isin(train["EntryID"].unique())
                ]
                if not args.one_vs_all:
                    stringdb_data = stringdb_data[
                        ~stringdb_data["subject_id"].isin(test["EntryID"].unique())
                    ]
                logger.info(f"Loaded {len(stringdb_data)} StringDB interactions")

            pairwise_alignment = pairwise_alignment[
                pairwise_alignment["query_id"].isin(test["EntryID"].unique())
                & pairwise_alignment["subject_id"].isin(train["EntryID"].unique())
            ]

            if not args.one_vs_all:
                logger.info(f"Removing test proteins from alignment subjects...")
                pairwise_alignment = pairwise_alignment[
                    ~pairwise_alignment["subject_id"].isin(test["EntryID"].unique())
                ]

            logger.info(f"Loaded {len(pairwise_alignment)} pairwise alignments")

            logger.info("Running alignment-based methods...")
            os.makedirs(f"{output_dir}/predictions/AlignmentScore", exist_ok=True)
            os.makedirs(f"{output_dir}/predictions/BlastKNN", exist_ok=True)
            os.makedirs(f"{output_dir}/predictions/IDScore", exist_ok=True)
            (
                unaligned_protein_ids,
                ascore_pred,
                blastknn_preds_dict,
                idscore_pred,
            ) = methods.transfer_annotations(
                logger,
                pairwise_alignment,
                train,
                test,
                args.k_values,
                one_vs_all=args.one_vs_all,
                stringdb_data=stringdb_data,
                stringdb_mode=stringdb_mode,
                stringdb_weight=stringdb_weight,
            )

            n_test = test["EntryID"].nunique()
            logger.info(
                f"Proteins with no alignment hit: {len(unaligned_protein_ids)} / {n_test} "
                f"({len(unaligned_protein_ids) / n_test * 100:.2f} %)"
            )

            unaligned_path = os.path.join(
                output_dir,
                f"unaligned_proteins_{args.dataset}_{db_version}_{aspect}.txt",
            )
            with open(unaligned_path, "w") as f:
                for pid in unaligned_protein_ids:
                    f.write(f"{pid}\n")

            if idscore_pred:
                pd.DataFrame(idscore_pred).to_csv(
                    f"{output_dir}/predictions/IDScore/predictions.tsv",
                    sep="\t",
                    index=False,
                )
                logger.info(
                    f"Saved {len(idscore_pred)} Best identity % score predictions"
                )
            else:
                logger.warning("No IDScore predictions were made.")

            if ascore_pred:
                pd.DataFrame(ascore_pred).to_csv(
                    f"{output_dir}/predictions/AlignmentScore/predictions.tsv",
                    sep="\t",
                    index=False,
                )
                logger.info(f"Saved {len(ascore_pred)} AlignmentScore predictions")
            else:
                logger.warning("No AlignmentScore predictions were made.")

            for k in args.k_values:
                pred_count = len(blastknn_preds_dict[k])
                if pred_count != 0:
                    pd.DataFrame(blastknn_preds_dict[k]).to_csv(
                        f"{output_dir}/predictions/BlastKNN/k{k}_predictions.tsv",
                        sep="\t",
                        index=False,
                    )
                    logger.info(f"Saved {pred_count} BlastKNN predictions for k={k}")
                else:
                    logger.warning(f"No BlastKNN predictions for k={k}")

            logger.info(f"All predictions saved to {output_dir}/predictions")
            logger.info(f"Completed processing for {aspect}")

            logger.info("Evaluating predictions...")

            evaluation.evaluate(
                logger, output_dir, args.dataset, aspect, k_values=args.k_values
            )
            logger.info(f"Evaluation completed for aspect {aspect}")

        print("Done!")


if __name__ == "__main__":
    main()

# Usage:
# python main.py --dataset ATGO \
# --alignment_dir ./data/swissprot/2024_01/diamond_swissprot_2024_01_alignment.tsv --k_values 1 3 5 10 15 20 \
# --aspects BPO CCO MFO --experimental_only
