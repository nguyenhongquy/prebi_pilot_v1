from __future__ import annotations

from itertools import combinations, product
from typing import Any

import krippendorff
import numpy as np
import pandas as pd


def encode_score(score: float, band_mapping: dict[str, str] | None = None) -> str:
    value = float(score)
    if not np.isfinite(value) or not 0 <= value <= 3 or not np.isclose(value * 10, round(value * 10)):
        raise ValueError("Scores must be finite values from 0 to 3 in steps of 0.1.")
    label = f"{value:.1f}"
    if band_mapping is None:
        return label
    if label not in band_mapping or band_mapping[label] not in {"0", "1", "2", "3"}:
        raise ValueError("An explicit mapping to bands 0-3 is required for every score.")
    return band_mapping[label]


def consistency_tables(
    scores: pd.DataFrame,
    *,
    units: pd.DataFrame,
    repetitions: int = 5,
    band_mapping: dict[str, str] | None = None,
    conditions: tuple[str, ...] = ("G1", "G2", "G3"),
) -> dict[str, pd.DataFrame]:
    if repetitions < 2:
        raise ValueError("At least two repetitions are required.")
    if len(conditions) < 2 or len(set(conditions)) != len(conditions) or any(not condition.strip() for condition in conditions):
        raise ValueError("At least two unique condition names are required.")
    required = {"document_id", "condition", "dimension_id", "repetition", "score"}
    if not required.issubset(scores.columns):
        raise ValueError("Missing repetition score columns.")
    if not {"document_id", "mode", "author_id"}.issubset(units.columns) or units.document_id.duplicated().any():
        raise ValueError("Units require unique document IDs, mode, and author ID.")
    if units[["document_id", "mode", "author_id"]].isna().any().any():
        raise ValueError("Unit identifiers and cohort labels must not be missing.")
    if scores.duplicated(["document_id", "condition", "dimension_id", "repetition"]).any():
        raise ValueError("Duplicate repetition ratings must not be counted twice.")
    if not scores.condition.isin(conditions).all() or not scores.dimension_id.isin(["SW", "UA", "HA"]).all():
        raise ValueError("Unknown condition or dimension.")
    if not scores.repetition.isin(range(1, repetitions + 1)).all():
        raise ValueError("Repetition IDs must be within the frozen protocol.")
    if not set(scores.document_id).issubset(set(units.document_id)):
        raise ValueError("Ratings refer to documents outside the frozen cohort.")
    ordered = [str(band) for band in range(4)] if band_mapping is not None else [f"{step / 10:.1f}" for step in range(31)]
    if band_mapping is not None:
        labels = [encode_score(step / 10, band_mapping) for step in range(31)]
        if labels != sorted(labels, key=int):
            raise ValueError("The categorical mapping must preserve score order.")
    frame = scores.copy()
    frame["category"] = [encode_score(score, band_mapping) for score in frame.score.tolist()]
    frame = frame.merge(units, on="document_id", validate="many_to_one")
    within: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    between: list[dict[str, Any]] = []
    category_indices = {label: index for index, label in enumerate(ordered)}
    for mode, cohort in units.groupby("mode", sort=True):
        document_ids = cohort.document_id.tolist()
        authors_by_document = dict(zip(cohort.document_id.tolist(), cohort.author_id.tolist(), strict=True))
        for dimension in ("SW", "UA", "HA"):
            tables = {}
            for condition in conditions:
                subset = frame.loc[
                    frame["mode"].eq(mode) & frame.dimension_id.eq(dimension) & frame.condition.eq(condition)
                ]
                table = subset.pivot(index="document_id", columns="repetition", values="category").reindex(
                    index=document_ids, columns=range(1, repetitions + 1),
                )
                tables[condition] = table
                complete = table.dropna()
                pair_rates = []
                for document_id, row in table.iterrows():
                    observed = row.dropna().tolist()
                    pairs = list(combinations(observed, 2))
                    rate = float(np.mean([left == right for left, right in pairs])) if pairs else None
                    is_complete = len(observed) == repetitions
                    if is_complete:
                        pair_rates.append(rate)
                    details.append({
                        "mode": mode, "condition": condition, "dimension_id": dimension,
                        "document_id": document_id, "author_id": authors_by_document[document_id],
                        "valid_repetitions": len(observed), "complete": is_complete,
                        "pairwise_exact": rate,
                        "unanimous": len(set(observed)) == 1 if is_complete else None,
                    })
                numeric = complete.map(lambda label: category_indices[label]).to_numpy(dtype=float).T
                alpha_results = {}
                for level in ("nominal", "ordinal"):
                    alpha = None
                    reason = None
                    if numeric.size == 0:
                        reason = "no_complete_units"
                    elif len(np.unique(numeric)) < 2:
                        reason = "no_category_variation"
                    else:
                        alpha = float(krippendorff.alpha(
                            reliability_data=numeric, value_domain=np.arange(len(ordered)),
                            level_of_measurement=level,
                        ))
                        if not np.isfinite(alpha):
                            alpha, reason = None, "undefined"
                    alpha_results[f"alpha_{level}"] = alpha
                    alpha_results[f"alpha_{level}_status"] = reason or "defined"
                within.append({
                    "mode": mode, "condition": condition, "dimension_id": dimension,
                    "documents": len(table), "complete_documents": len(complete),
                    "missing_ratings": int(table.isna().sum().sum()),
                    "pairwise_exact": float(np.mean(pair_rates)) if pair_rates else None,
                    "unanimous_rate": float(complete.nunique(axis=1).eq(1).mean()) if len(complete) else None,
                    **alpha_results,
                })
            for left, right in combinations(conditions, 2):
                rates = []
                for document_id in document_ids:
                    left_values = tables[left].loc[document_id].dropna().tolist()
                    right_values = tables[right].loc[document_id].dropna().tolist()
                    if len(left_values) == repetitions and len(right_values) == repetitions:
                        rates.append(float(np.mean([first == second for first, second in product(left_values, right_values)])))
                between.append({
                    "mode": mode, "dimension_id": dimension, "left": left, "right": right,
                    "complete_documents": len(rates),
                    "all_pairs_exact": float(np.mean(rates)) if rates else None,
                })
    frequencies = frame.groupby(["mode", "condition", "dimension_id", "category"]).size().rename("count").reset_index()
    return {
        "within": pd.DataFrame(within), "per_document": pd.DataFrame(details),
        "between": pd.DataFrame(between), "frequencies": frequencies,
    }


def paired_consistency_differences(
    details: pd.DataFrame, *, samples: int = 2000, seed: int = 20260929,
    contrasts: tuple[tuple[str, str], ...] = (("G1", "G2"), ("G1", "G3"), ("G2", "G3")),
) -> pd.DataFrame:
    if samples < 1:
        raise ValueError("Bootstrap samples must be positive.")
    results = []
    random = np.random.default_rng(seed)
    for (mode, dimension), subset in details.groupby(["mode", "dimension_id"]):
        complete = subset.loc[subset["complete"]]
        table = complete.pivot(index=["document_id", "author_id"], columns="condition", values="pairwise_exact")
        for left, right in contrasts:
            if left not in table or right not in table:
                continue
            paired = table[[left, right]].dropna()
            differences = paired[right] - paired[left]
            authors = paired.index.get_level_values("author_id").unique().tolist()
            estimates = []
            if len(authors) >= 2:
                clusters = [differences.loc[paired.index.get_level_values("author_id") == author].to_numpy() for author in authors]
                for _ in range(samples):
                    sampled = random.integers(0, len(clusters), size=len(clusters))
                    estimates.append(float(np.concatenate([clusters[index] for index in sampled]).mean()))
            results.append({
                "mode": mode, "dimension_id": dimension, "contrast": f"{right}_minus_{left}",
                "documents": len(paired), "authors": len(authors),
                "difference": float(differences.mean()) if len(paired) else None,
                "lower_95": float(np.quantile(estimates, 0.025)) if estimates else None,
                "upper_95": float(np.quantile(estimates, 0.975)) if estimates else None,
                "interval_status": "exploratory_author_cluster_bootstrap" if estimates else "insufficient_author_clusters",
            })
    return pd.DataFrame(results)