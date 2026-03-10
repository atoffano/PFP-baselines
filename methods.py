import pandas as pd
from collections import defaultdict
import tqdm


def score(E, score_col="bit_score"):
    """
    Accumulate annotation term scores from neighbour rows.

    Parameters:
    E (dataframe): Rows with 'subject_annotations' and a numeric score column.
    score_col (str): Column to use as the per-row weight.
    """
    go_terms = defaultdict(float)
    for _, row in E.iterrows():
        subject_annotations = row["subject_annotations"]
        s = row[score_col]
        for annotation in subject_annotations:
            go_terms[annotation] += s
    return go_terms


def alignment_score(E, score_col="bit_score"):
    """
    Compute normalised weighted annotation score.

    Parameters:
    E (dataframe): Rows with 'subject_annotations' and a numeric score column.
    score_col (str): Column to use as the per-row weight (default: 'bit_score').
    """
    go_terms = score(E, score_col)
    total = E[score_col].sum()
    if total > 0:
        for key in go_terms:
            go_terms[key] /= total
    return go_terms


def alignment_knn(E, k=5, score_col="bit_score"):
    """
    Transfer annotations from the k most similar proteins.

    Parameters:
    E (dataframe): Rows with 'subject_annotations' and a numeric score column.
    k (int): Number of nearest neighbours to consider.
    score_col (str): Column to sort and weight by (default: 'bit_score').
    """
    top_k = E.sort_values(by=score_col, ascending=False).head(k)
    go_terms = score(top_k, score_col)
    total_score = top_k[score_col].sum()
    if total_score > 0:
        for key in go_terms:
            go_terms[key] /= total_score
    return go_terms


def _merge_preds(pred_a, pred_b, weight):
    """
    Linearly blend two {term: score} dicts.
    merged[t] = (1 - weight) * pred_a[t] + weight * pred_b[t]
    Terms present in only one dict contribute a 0 for the other.
    """
    merged = {}
    for term in set(pred_a) | set(pred_b):
        merged[term] = (1 - weight) * pred_a.get(term, 0.0) + weight * pred_b.get(term, 0.0)
    return merged


def best_percent_identity(E):
    """
    Get the best percent identity from the alignments. Transfer its annotations to current protein.

    Parameters:
    E (dataframe): Dataframe of similar sequences according to Diamond blast results,
                  with columns 'subject_id', 'perc_identity'.
    """
    if E.empty:
        return {}

    # Get the row with the maximum percent identity
    best_row = E.loc[E["perc_identity"].idxmax()]
    annotations = best_row["subject_annotations"]
    go_terms = {term: 1.0 for term in annotations}  # Assign a score of 1.0 to each term

    return go_terms


def naive_baseline(input_dir, train, val):
    # Compute the frequency of each GO term across all proteins in the training set
    go_term_counts = train["term"].value_counts()
    go_term_scores = go_term_counts / train["EntryID"].nunique()

    # Assign the same scores to all query proteins
    query_proteins = val["EntryID"].unique()
    predictions = pd.DataFrame(
        [
            (protein, term, score)
            for protein in query_proteins
            for term, score in go_term_scores.items()
        ],
        columns=["target_ID", "term_ID", "score"],
    )
    predictions.to_csv(
        f"{input_dir}/predictions/NaiveBaseline/predictions.tsv",
        sep="\t",
        index=False,
    )


def transfer_annotations(
    logger,
    pairwise_alignment,
    train,
    test,
    k_values,
    one_vs_all=False,
    stringdb_data=None,
    stringdb_mode=None,
    stringdb_weight=0.5,
):
    """
    Parameters
    ----------
    stringdb_data : pd.DataFrame or None
        StringDB interactions with columns query_id, subject_id, combined_score (0-1).
        When provided, stringdb_mode controls how it is combined with alignment predictions.
    stringdb_mode : str or None
        'rescue' – use StringDB predictions only for proteins with no alignment hits.
        'merge'  – linearly blend alignment and StringDB predictions for all proteins.
    stringdb_weight : float
        Weight applied to StringDB predictions in 'merge' mode (0-1). Default 0.5.
    """
    # Annotations for each sequence in the known protein set (used to transfer annotations)
    # Ts: A dictionary of annotations like {'protein_name': ['go_term1', 'go_term2', ...], ...}
    Ts = train.groupby("EntryID")["term"].apply(list).to_dict()

    pairwise_alignment["subject_annotations"] = pairwise_alignment["subject_id"].map(Ts)
    pairwise_alignment = pairwise_alignment[
        pairwise_alignment["subject_annotations"].map(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        > 0
    ]  # Drop rows without annotations

    grouped = pairwise_alignment.groupby(
        "query_id"
    )  # Group by test protein for faster processing

    # Prepare StringDB lookup if provided
    stringdb_grouped = None
    if stringdb_data is not None and stringdb_mode is not None:
        stringdb_data = stringdb_data.copy()
        stringdb_data["subject_annotations"] = stringdb_data["subject_id"].map(Ts)
        stringdb_data = stringdb_data[
            stringdb_data["subject_annotations"].map(
                lambda x: len(x) if isinstance(x, list) else 0
            ) > 0
        ]
        stringdb_grouped = stringdb_data.groupby("query_id")

    ascore_pred, idscore_pred = [], []
    blastknn_preds_dict = {k: [] for k in k_values}

    unaligned_proteins = 0
    unaligned_protein_ids = []  # Store unannotated protein IDs
    for protein in tqdm.tqdm(
        test["EntryID"].unique(), desc="Computing Diamond-based predictions"
    ):
        # ── Alignment-based predictions ──────────────────────────────────────
        has_alignment = True
        try:
            group = grouped.get_group(protein)
        except KeyError:  # No alignments for this protein
            has_alignment = False
            unaligned_proteins += 1
            unaligned_protein_ids.append(protein)

        if has_alignment and not one_vs_all:
            try:
                assert (
                    not group["subject_id"].isin(test["EntryID"].unique()).any()
                ), "Annotation leakage has been found beetween protein sets !"
            except AssertionError as e:
                logger.warning(f"Warning for protein {protein}: {e}")
                logger.warning(
                    f"Leakage in:\n{group[group['subject_id'].isin(test['EntryID'].unique())]}"
                )
                exit(1)

        if has_alignment:
            aln_ascore = alignment_score(group)
            aln_idscore = best_percent_identity(group)
            aln_knn = {k: alignment_knn(group, k=k) for k in k_values}
        else:
            aln_ascore, aln_idscore, aln_knn = {}, {}, {k: {} for k in k_values}

        # ── StringDB predictions ─────────────────────────────────────────────
        sdb_ascore, sdb_idscore, sdb_knn = {}, {}, {k: {} for k in k_values}
        if stringdb_grouped is not None:
            # Only compute StringDB if needed (rescue: unaligned only; merge: always)
            if stringdb_mode == "rescue" and not has_alignment or stringdb_mode == "merge":
                try:
                    sdb_group = stringdb_grouped.get_group(protein)
                    sdb_ascore = alignment_score(sdb_group, score_col="combined_score")
                    sdb_idscore = {
                        term: row["combined_score"]
                        for _, row in [next(iter(sdb_group.sort_values("combined_score", ascending=False).iterrows()))]
                        for term in row["subject_annotations"]
                    }
                    sdb_knn = {k: alignment_knn(sdb_group, k=k, score_col="combined_score") for k in k_values}
                except (KeyError, StopIteration):
                    pass  # No StringDB neighbours for this protein

        # ── Combine predictions ──────────────────────────────────────────────
        if stringdb_mode == "rescue" and not has_alignment:
            # Use StringDB predictions directly for proteins with no alignment
            final_ascore = sdb_ascore
            final_idscore = sdb_idscore
            final_knn = sdb_knn
        elif stringdb_mode == "merge":
            final_ascore = _merge_preds(aln_ascore, sdb_ascore, stringdb_weight)
            final_idscore = _merge_preds(aln_idscore, sdb_idscore, stringdb_weight)
            final_knn = {k: _merge_preds(aln_knn[k], sdb_knn[k], stringdb_weight) for k in k_values}
        else:
            if not has_alignment:
                continue  # No predictions and no StringDB fallback
            final_ascore = aln_ascore
            final_idscore = aln_idscore
            final_knn = aln_knn

        # ── Emit predictions ─────────────────────────────────────────────────
        if final_idscore:
            idscore_pred.extend(
                {"target_ID": protein, "term_ID": term_id, "score": sc}
                for term_id, sc in final_idscore.items()
            )

        for k in k_values:
            blastknn_preds_dict[k].extend(
                {"target_ID": protein, "term_ID": term_id, "score": sc}
                for term_id, sc in final_knn[k].items()
            )

        ascore_pred.extend(
            {"target_ID": protein, "term_ID": term_id, "score": sc}
            for term_id, sc in final_ascore.items()
        )
    logger.info(
        f"Number of unaligned proteins: {unaligned_proteins} out of {len(test['EntryID'].unique())} ({unaligned_proteins / test['EntryID'].nunique() * 100} %); No annotations have been transfered for alignment-based methods."
    )
    return unaligned_protein_ids, ascore_pred, blastknn_preds_dict, idscore_pred
