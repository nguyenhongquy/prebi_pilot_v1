from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from reflection_assessment_feedback.experiment_config import PROJECT_ROOT, load_experiment_config

REMAINING_TEST_COHORT = "remaining-test-reference-complete-v1"

GOLD_SCOPE_COLUMN = "scope_target"
GOLD_SCOPE_KNOWN_COLUMN = "scope_target_known"
GOLD_REVIEW_COLUMN = "requires_review"
GOLD_DIMENSIONS = {
    "SW": ("target_situationserfassung", "known_situationserfassung"),
    "UA": ("target_analyse", "known_analyse"),
    "HA": ("target_konsequenzen", "known_konsequenzen"),
}


def build_gold_segment_analysis_summary(
    modeling_wide_path: str | Path,
    document_id: str,
) -> dict[str, Any]:
    required_columns = {
        "document_id",
        "segment_id",
        GOLD_SCOPE_COLUMN,
        GOLD_SCOPE_KNOWN_COLUMN,
        GOLD_REVIEW_COLUMN,
        *(column for columns in GOLD_DIMENSIONS.values() for column in columns),
    }


    rows = pd.read_csv(
        modeling_wide_path,
        usecols=sorted(required_columns),
        dtype=str,
    )
    rows["document_id"] = rows["document_id"].fillna("").str.strip()
    rows = rows.loc[rows["document_id"] == str(document_id)].copy()
    if rows.empty:
        raise ValueError("No gold segment annotations found for the judge document.")
    if rows["segment_id"].isna().any() or rows["segment_id"].str.strip().eq("").any():
        raise ValueError("Gold segment annotations contain missing segment IDs.")

    rows[GOLD_SCOPE_COLUMN] = rows[GOLD_SCOPE_COLUMN].fillna("").str.strip()
    rows[GOLD_SCOPE_KNOWN_COLUMN] = rows[GOLD_SCOPE_KNOWN_COLUMN].fillna("").str.strip().str.lower()
    rows[GOLD_REVIEW_COLUMN] = rows[GOLD_REVIEW_COLUMN].fillna("").str.strip().str.lower()
    scope_known = rows[GOLD_SCOPE_KNOWN_COLUMN].isin({"1", "true", "yes"})
    invalid_scope = scope_known & ~rows[GOLD_SCOPE_COLUMN].isin({"0", "1"})
    if invalid_scope.any():
        raise ValueError("A known gold scope value is not encoded as 0 or 1.")
    in_scope = scope_known & rows[GOLD_SCOPE_COLUMN].eq("1")
    out_of_scope = scope_known & rows[GOLD_SCOPE_COLUMN].eq("0")

    scope_counts = {
        "in_scope": int(in_scope.sum()),
        "out_of_scope": int(out_of_scope.sum()),
        "unknown": int((~scope_known).sum()),
    }
    dimension_summaries: dict[str, dict[str, Any]] = {}
    for dimension_id, (target_column, known_column) in GOLD_DIMENSIONS.items():
        rows[target_column] = rows[target_column].fillna("").str.strip()
        rows[known_column] = rows[known_column].fillna("").str.strip().str.lower()
        known_band = rows[known_column].isin({"1", "true", "yes"})
        valid_band = rows[target_column].isin({"0", "1", "2", "3"})
        invalid_known_band = in_scope & known_band & ~valid_band
        if invalid_known_band.any():
            raise ValueError(f"Known gold component bands are invalid for {dimension_id}.")
        usable_band = in_scope & known_band & valid_band
        dimension_summaries[dimension_id] = {
            "known_in_scope_count": int(usable_band.sum()),
            "unknown_in_scope_count": int((in_scope & ~usable_band).sum()),
            "band_counts": {
                band: int((usable_band & rows[target_column].eq(band)).sum())
                for band in ("0", "1", "2", "3")
            },
        }

    review_required = rows[GOLD_REVIEW_COLUMN].isin({"1", "true", "yes"})
    unique_segments = rows["segment_id"].nunique()
    candidate_counts = rows.groupby("segment_id").size()
    return {
        "summary_version": "1.0.0",
        "analysis_source": "provisional_gold_candidate_annotations",
        "aggregation_unit": "candidate_annotation",
        "segment_count": int(unique_segments),
        "candidate_annotation_count": int(len(rows)),
        "segments_with_multiple_candidates": int((candidate_counts > 1).sum()),
        "candidate_annotations_requiring_review": int(review_required.sum()),
        "scope_candidate_counts": scope_counts,
        "component_band_summaries": dimension_summaries,
    }


def load_author_document_group(
    split_manifest_path: str | Path,
    target_document_id: str,
) -> pd.DataFrame:
    manifest = pd.read_csv(
        split_manifest_path,
        usecols=["document_id", "author_id", "split"],
        dtype={"document_id": str, "author_id": str, "split": str},
    )
    for column in ("document_id", "author_id", "split"):
        manifest[column] = manifest[column].fillna("").str.strip()
    target_rows = manifest.loc[manifest["document_id"] == str(target_document_id)]
    authors = set(target_rows["author_id"])
    if len(target_rows) != 1 or len(authors) != 1 or not next(iter(authors)):
        raise ValueError("The target document must map to exactly one author group.")
    author_id = next(iter(authors))
    author_documents = manifest.loc[manifest["author_id"] == author_id].copy()
    if author_documents["document_id"].duplicated().any():
        raise ValueError("Author-group manifest contains duplicate document IDs.")
    if (author_documents["split"] != "test").any():
        raise ValueError("The playground author group must be wholly contained in the test split.")
    return author_documents.sort_values("document_id").reset_index(drop=True)


def load_remaining_test_feedback_cohort() -> pd.DataFrame:
    config = load_experiment_config()
    dataset_config = config["dataset"]
    if dataset_config.get("generation_cohort") != REMAINING_TEST_COHORT:
        raise ValueError("Experiment config does not select the reference-complete remaining-test cohort.")

    split_path = (
        PROJECT_ROOT
        / config["paths"]["provisional_split_directory"]
        / f"provisional_{dataset_config['prediction_split']}_modeling_wide.csv"
    )
    feedback_path = (PROJECT_ROOT / config["paths"]["human_feedback"]).resolve()
    if not split_path.is_file() or not feedback_path.is_file():
        raise FileNotFoundError("The configured test split or teacher-feedback source is unavailable.")

    split_ids = set(
        pd.read_csv(split_path, usecols=["document_id"], dtype={"document_id": str})[
            "document_id"
        ].dropna().astype(str).str.strip()
    )
    feedback = pd.read_csv(
        feedback_path,
        usecols=["essay_name", "document_id", "full_text", "feedback_text"],
        dtype={"essay_name": str, "document_id": str},
    )
    feedback["essay_name"] = feedback["essay_name"].fillna("").str.strip().str.upper()
    feedback["document_id"] = feedback["document_id"].fillna("").str.strip()
    feedback["full_text"] = feedback["full_text"].fillna("").astype(str).str.strip()
    feedback["feedback_text"] = feedback["feedback_text"].fillna("").astype(str).str.strip()

    development_documents = load_author_document_group(
        split_path.parent / "provisional_split_manifest.csv",
        str(dataset_config["exploratory_document_id"]),
    )
    excluded_document_ids = {
        str(document_id) for document_id in dataset_config["generation_excluded_document_ids"]
    }
    selected_document_ids = (
        split_ids - set(development_documents["document_id"]) - excluded_document_ids
    )
    candidate_rows = feedback.loc[
        feedback["document_id"].isin(selected_document_ids)
    ].copy()

    eligible_records: list[dict[str, Any]] = []
    for document_id in sorted(selected_document_ids):
        rows = candidate_rows.loc[candidate_rows["document_id"].eq(document_id)]
        full_texts = {value for value in rows["full_text"] if value}
        feedback_texts = {value for value in rows["feedback_text"] if value}
        if len(rows) != 1 or len(full_texts) != 1 or len(feedback_texts) != 1:
            raise ValueError(
                f"Selected document {document_id} lacks a unique complete teacher-reference record."
            )
        eligible_records.append(rows.iloc[0].to_dict())

    expected_count = dataset_config.get("expected_generation_document_count")
    if expected_count is None or len(eligible_records) != expected_count:
        raise ValueError(
            "Reference-complete remaining-test cohort size differs from experiment config."
        )
    return pd.DataFrame.from_records(eligible_records).sort_values("document_id").reset_index(drop=True)