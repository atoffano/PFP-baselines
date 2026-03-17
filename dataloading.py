import pandas as pd
from constants import *


def load_uniprot_mapping():
    """
    Map Uniprot IDs in the DataFrame using the provided id_mapping dictionary.
    """
    id_mapping = pd.read_csv(
        f"./data/swissprot/2024_01/swissprot_2024_01_annotations.tsv",  # Most up to date mapping
        sep="\t",
        usecols=["EntryID", "Entry Name"],
    )
    id_mapping = id_mapping.set_index("Entry Name").to_dict()["EntryID"]
    return id_mapping


def load_pairwise_alignment(dataset, id_mapping=None):
    """
    Load pairwise alignment data (SwissProt 2024_01) and map Query_id and Subject_id to EntryID using id_mapping
    """
    pairwise_alignment = pd.read_csv(
        "./data/swissprot/2024_01/diamond_swissprot_2024_01_alignment.tsv",
        sep="\t",
        header=None,
        names=[
            "query_id",
            "subject_id",
            "perc_identity",
            "align_length",
            "mismatches",
            "gap_opens",
            "q_start",
            "q_end",
            "s_start",
            "s_end",
            "e_value",
            "bit_score",
        ],
    )

    # Load Uniprot ID mapping
    if dataset in USES_ENTRYID:
        # Diamond output uses EntryName (e.g. Q6GZX1) as protein IDs
        pairwise_alignment["query_id"] = pairwise_alignment["query_id"].map(id_mapping)
        pairwise_alignment["subject_id"] = pairwise_alignment["subject_id"].map(
            id_mapping
        )

    # Drop rows with NaN values in Query_id or Subject_id
    pairwise_alignment = pairwise_alignment[
        pairwise_alignment["query_id"].notna()
        & pairwise_alignment["subject_id"].notna()
    ]

    # Remove self-alignments
    # This is required to avoid self-annotation transfer
    pairwise_alignment = pairwise_alignment[
        pairwise_alignment["query_id"] != pairwise_alignment["subject_id"]
    ]

    return pairwise_alignment


def load_stringdb(dataset, id_mapping=None):
    """
    Load STRING DB interactions for SwissProt 2024_01 and map protein IDs to UniProt accessions.
    Returns a DataFrame with columns: query_id, subject_id, combined_score (normalised 0-1).
    """
    # Build StringDB ID -> UniProt accession reverse mapping
    id_map_df = pd.read_csv(
        "./data/swissprot/2024_01/idmapping_swissprot_stringdb.tsv",
        sep="\t",
    )
    stringdb_to_uniprot = dict(zip(id_map_df["To"], id_map_df["From"]))

    # Load interactions
    stringdb = pd.read_csv(
        "./data/swissprot/2024_01/swissprot_stringdb.tsv",
        sep="\t",
        usecols=["protein1", "protein2", "combined_score"],
    )

    # Map StringDB IDs to UniProt accessions
    stringdb["query_id"] = stringdb["protein1"].map(stringdb_to_uniprot)
    stringdb["subject_id"] = stringdb["protein2"].map(stringdb_to_uniprot)
    stringdb = stringdb.dropna(subset=["query_id", "subject_id"])
    stringdb = stringdb[stringdb["query_id"] != stringdb["subject_id"]]

    # Normalise combined_score from 0-1000 to 0-1
    stringdb["combined_score"] = stringdb["combined_score"] / 1000.0

    # STRING DB stores each pair only once (A→B, not B→A).
    # Symmetrize so that query/subject filters work regardless of storage direction.
    mirrored = stringdb.rename(
        columns={"query_id": "subject_id", "subject_id": "query_id"}
    )[["query_id", "subject_id", "combined_score"]]
    stringdb = pd.concat(
        [stringdb[["query_id", "subject_id", "combined_score"]], mirrored],
        ignore_index=True,
    ).drop_duplicates()

    # For datasets that use mnemonic EntryIDs (e.g. CAFA3, D1), remap accessions
    if dataset in USES_ENTRYID and id_mapping is not None:
        stringdb["query_id"] = stringdb["query_id"].map(id_mapping)
        stringdb["subject_id"] = stringdb["subject_id"].map(id_mapping)
        stringdb = stringdb.dropna(subset=["query_id", "subject_id"])

    return stringdb[["query_id", "subject_id", "combined_score"]]


def load_data(
    logger,
    dataset,
    aspect,
    db_version,
    annotations_2024_01=None,
    id_mapping=None,
    experimental_only=False,
    one_vs_all=False,
):
    logger.info(f"Loading SwissProt {db_version} annotations for training set")
    if db_version == "":
        # Load constrained dataset
        train = pd.read_csv(
            f"./data/{dataset}/{dataset}_{aspect}_train_annotations.tsv",
            sep="\t",
        )
    elif experimental_only:
        # Load experimental annotations only
        train = pd.read_csv(
            f"./data/swissprot/{db_version}/swissprot_{db_version}_{aspect}_exp_annotations.tsv",
            sep="\t",
        )
    else:
        train = pd.read_csv(
            f"./data/swissprot/{db_version}/swissprot_{db_version}_{aspect}_annotations.tsv",
            sep="\t",
        )

    # Proteins to annotate
    test = pd.read_csv(
        f"./data/{dataset}/{dataset}_{aspect}_test_annotations.tsv",
        sep="\t",
    )

    if not one_vs_all:
        test = test[["EntryID"]]  # Drop terms; Makes sure no leakage occurs
        train = train[
            ~train["EntryID"].isin(test["EntryID"].unique())
        ]  # To *really* make sure

    if annotations_2024_01:
        logger.info("Fixing train proteins' annotations to 2024 SwissProt version...")
        if experimental_only:
            annotations_2024_01 = pd.read_csv(
                f"./data/swissprot/2024_01/swissprot_2024_01_{aspect}_exp_annotations.tsv",
                sep="\t",
            )
        else:
            annotations_2024_01 = pd.read_csv(
                f"./data/swissprot/2024_01/swissprot_2024_01_{aspect}_annotations.tsv",
                sep="\t",
            )
        # Get rows in annotations_2024_01 where EntryID is in train
        train = annotations_2024_01[
            annotations_2024_01["EntryID"].isin(train["EntryID"])
        ]

    if dataset in USES_ENTRYID:
        logger.info("Mapping train SwissProt EntryID to EntryName...")
        train["EntryID"] = train["EntryID"].map(id_mapping).fillna(train["EntryID"])

    train["term"] = train["term"].str.split("; ").dropna()
    train = train.explode("term").drop_duplicates()

    return train, test
