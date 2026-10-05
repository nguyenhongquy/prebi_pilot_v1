from __future__ import annotations

import csv

import pytest

from reflection_assessment_feedback.development_cohorts import (
    build_gold_segment_analysis_summary,
    load_author_document_group,
)


def test_gold_analysis_summary_preserves_candidate_ambiguity_and_missingness(tmp_path) -> None:
    path = tmp_path / "gold.csv"
    fields = [
        "document_id", "segment_id", "scope_target", "scope_target_known", "requires_review",
        "target_situationserfassung", "known_situationserfassung",
        "target_analyse", "known_analyse",
        "target_konsequenzen", "known_konsequenzen",
    ]
    rows = [
        {
            "document_id": "doc-a", "segment_id": "seg-1", "scope_target": "1",
            "scope_target_known": "1", "requires_review": "0",
            "target_situationserfassung": "0", "known_situationserfassung": "1",
            "target_analyse": "2", "known_analyse": "1",
            "target_konsequenzen": "", "known_konsequenzen": "0",
        },
        {
            "document_id": "doc-a", "segment_id": "seg-1", "scope_target": "0",
            "scope_target_known": "1", "requires_review": "1",
            "target_situationserfassung": "N", "known_situationserfassung": "1",
            "target_analyse": "N", "known_analyse": "1",
            "target_konsequenzen": "N", "known_konsequenzen": "1",
        },
        {
            "document_id": "doc-a", "segment_id": "seg-2", "scope_target": "",
            "scope_target_known": "0", "requires_review": "0",
            "target_situationserfassung": "", "known_situationserfassung": "0",
            "target_analyse": "", "known_analyse": "0",
            "target_konsequenzen": "", "known_konsequenzen": "0",
        },
        {
            "document_id": "doc-b", "segment_id": "other", "scope_target": "1",
            "scope_target_known": "1", "requires_review": "0",
            "target_situationserfassung": "3", "known_situationserfassung": "1",
            "target_analyse": "3", "known_analyse": "1",
            "target_konsequenzen": "3", "known_konsequenzen": "1",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = build_gold_segment_analysis_summary(path, "doc-a")

    assert summary["analysis_source"] == "provisional_gold_candidate_annotations"
    assert summary["aggregation_unit"] == "candidate_annotation"
    assert summary["segment_count"] == 2
    assert summary["candidate_annotation_count"] == 3
    assert summary["segments_with_multiple_candidates"] == 1
    assert summary["candidate_annotations_requiring_review"] == 1
    assert summary["scope_candidate_counts"] == {
        "in_scope": 1,
        "out_of_scope": 1,
        "unknown": 1,
    }
    assert summary["component_band_summaries"]["UA"]["band_counts"] == {
        "0": 0, "1": 0, "2": 1, "3": 0,
    }
    assert summary["component_band_summaries"]["UA"]["unknown_in_scope_count"] == 0
    assert summary["component_band_summaries"]["HA"]["unknown_in_scope_count"] == 1


def test_gold_analysis_summary_rejects_invalid_known_scope(tmp_path) -> None:
    path = tmp_path / "invalid.csv"
    fields = [
        "document_id", "segment_id", "scope_target", "scope_target_known", "requires_review",
        "target_situationserfassung", "known_situationserfassung",
        "target_analyse", "known_analyse",
        "target_konsequenzen", "known_konsequenzen",
    ]
    row = {
        "document_id": "doc-a", "segment_id": "seg-1", "scope_target": "unknown",
        "scope_target_known": "1", "requires_review": "0",
        "target_situationserfassung": "1", "known_situationserfassung": "1",
        "target_analyse": "1", "known_analyse": "1",
        "target_konsequenzen": "1", "known_konsequenzen": "1",
    }
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(ValueError, match="known gold scope"):
        build_gold_segment_analysis_summary(path, "doc-a")


def test_author_group_selector_returns_all_test_documents_and_rejects_split_overlap(tmp_path) -> None:
    path = tmp_path / "split.csv"
    fields = ["document_id", "author_id", "split"]
    rows = [
        {"document_id": "188", "author_id": "author-x", "split": "test"},
        {"document_id": "189", "author_id": "author-x", "split": "test"},
        {"document_id": "200", "author_id": "author-y", "split": "test"},
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    selected = load_author_document_group(path, "188")
    assert selected["document_id"].tolist() == ["188", "189"]

    rows[1]["split"] = "train"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="wholly contained in the test split"):
        load_author_document_group(path, "188")
