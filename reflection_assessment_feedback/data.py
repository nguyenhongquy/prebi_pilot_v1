from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from reflection_assessment_feedback.models import (
    DIMENSION_IDS,
    GenerationRequest,
    IntermediateAnalysis,
    ReflectionDocument,
    ReflectionSegment,
    Rubric,
    SegmentAnalysis,
)

TARGET_COLUMNS = {
    "SW": "target_situationserfassung",
    "UA": "target_analyse",
    "HA": "target_konsequenzen",
}
KNOWN_COLUMNS = {
    "SW": "known_situationserfassung",
    "UA": "known_analyse",
    "HA": "known_konsequenzen",
}
REQUIRED_COLUMNS = {
    "segment_id",
    "candidate_id",
    "text",
    "document_id",
    "segment_order",
    "scope_target",
    "scope_target_known",
    "requires_review",
    *TARGET_COLUMNS.values(),
    *KNOWN_COLUMNS.values(),
}


def _is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _read_document_rows(csv_path: str | Path, document_id: str) -> list[dict[str, str]]:
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
        rows = [row for row in reader if row["document_id"] == document_id]
    if not rows:
        raise ValueError("No rows found for the requested document ID.")
    return rows


def load_human_annotated_document(
    csv_path: str | Path,
    document_id: str,
) -> tuple[ReflectionDocument, IntermediateAnalysis]:
    rows = _read_document_rows(csv_path, document_id)
    seen_segments: set[str] = set()
    segments: list[ReflectionSegment] = []
    annotations: list[SegmentAnalysis] = []

    for row in rows:
        segment_id = row["segment_id"]
        if segment_id in seen_segments:
            raise ValueError(
                "The requested document has alternative candidate annotations. "
                "Use an adjudicated, one-candidate-per-segment input for G3."
            )
        seen_segments.add(segment_id)
        if _is_true(row["requires_review"]):
            raise ValueError("The requested document contains a segment requiring review.")
        if not _is_true(row["scope_target_known"]):
            raise ValueError("The requested document contains an unknown scope label.")

        scope_value = row["scope_target"].strip()
        if scope_value not in {"0", "1"}:
            raise ValueError("Scope labels must be encoded as 0 or 1.")
        in_scope = scope_value == "1"
        component_bands: dict[str, str] = {}
        for dimension_id in DIMENSION_IDS:
            if not in_scope:
                component_bands[dimension_id] = "N"
                continue
            target = row[TARGET_COLUMNS[dimension_id]].strip()
            if not _is_true(row[KNOWN_COLUMNS[dimension_id]]) or target not in {
                "0",
                "1",
                "2",
                "3",
            }:
                raise ValueError("The requested document has an unknown component label.")
            component_bands[dimension_id] = target

        segments.append(
            ReflectionSegment(
                segment_id=segment_id,
                text=row["text"].strip(),
                order=int(row["segment_order"]),
            )
        )
        annotations.append(
            SegmentAnalysis(
                segment_id=segment_id,
                scope="in_scope" if in_scope else "out_of_scope",
                component_bands=component_bands,
            )
        )

    segments.sort(key=lambda segment: segment.order)
    analysis_by_id = {annotation.segment_id: annotation for annotation in annotations}
    analysis = IntermediateAnalysis(
        source="human",
        segments=[analysis_by_id[segment.segment_id] for segment in segments],
    )
    return ReflectionDocument(document_id=document_id, segments=segments), analysis


def eligible_document_ids(csv_path: str | Path) -> list[str]:
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if "document_id" not in (reader.fieldnames or ()):
            raise ValueError("Input CSV must include document_id.")
        document_ids = sorted({row["document_id"] for row in reader if row["document_id"]})

    eligible: list[str] = []
    for document_id in document_ids:
        try:
            load_human_annotated_document(csv_path, document_id)
        except (ValueError, KeyError):
            continue
        eligible.append(document_id)
    return eligible

def make_generation_request(
    *,
    condition: str,
    document: ReflectionDocument,
    rubric: Rubric,
    analysis: IntermediateAnalysis | None = None,
) -> GenerationRequest:
    return GenerationRequest(
        condition=condition,
        document=document,
        rubric=rubric,
        analysis=analysis,
    )