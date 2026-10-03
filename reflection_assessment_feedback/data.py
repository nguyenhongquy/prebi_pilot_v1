from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from reflection_assessment_feedback.models import (
    DIMENSION_IDS,
    GenerationRequest,
    HumanAnalysisCandidate,
    HumanAnalysisContext,
    HumanSegmentAnalysis,
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


def load_human_analysis_context(
    csv_path: str | Path,
    document_id: str,
) -> tuple[ReflectionDocument, HumanAnalysisContext]:
    rows = _read_document_rows(csv_path, document_id)
    rows_by_segment: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_segment[row["segment_id"]].append(row)

    document_segments: list[ReflectionSegment] = []
    human_segments: list[HumanSegmentAnalysis] = []
    for segment_id, candidate_rows in rows_by_segment.items():
        texts = {row["text"].strip() for row in candidate_rows}
        orders = {int(row["segment_order"]) for row in candidate_rows}
        if len(texts) != 1 or len(orders) != 1:
            raise ValueError("Candidate copies disagree on segment text or source order.")
        document_segments.append(
            ReflectionSegment(
                segment_id=segment_id,
                text=next(iter(texts)),
                order=next(iter(orders)),
            )
        )

        candidates = []
        for row in candidate_rows:
            scope_value = row["scope_target"].strip()
            scope_known = _is_true(row["scope_target_known"])
            scope = (
                {"0": "out_of_scope", "1": "in_scope"}.get(scope_value)
                if scope_known
                else None
            )
            raw_component_values = {
                dimension_id: row[TARGET_COLUMNS[dimension_id]].strip()
                for dimension_id in DIMENSION_IDS
            }
            component_bands = {
                dimension_id: (
                    raw_component_values[dimension_id]
                    if raw_component_values[dimension_id] in {"N", "0", "1", "2", "3"}
                    else None
                )
                for dimension_id in DIMENSION_IDS
            }
            known_component_bands = {
                dimension_id: _is_true(row[KNOWN_COLUMNS[dimension_id]])
                for dimension_id in DIMENSION_IDS
            }
            candidates.append(
                HumanAnalysisCandidate(
                    candidate_id=row["candidate_id"].strip(),
                    scope=scope,
                    scope_known=scope_known,
                    raw_scope_value=scope_value,
                    component_bands=component_bands,
                    raw_component_values=raw_component_values,
                    known_component_bands=known_component_bands,
                    requires_review=_is_true(row["requires_review"]),
                )
            )
        human_segments.append(
            HumanSegmentAnalysis(segment_id=segment_id, candidates=candidates)
        )

    document_segments.sort(key=lambda segment: segment.order)
    human_by_id = {segment.segment_id: segment for segment in human_segments}
    document = ReflectionDocument(document_id=document_id, segments=document_segments)
    context = HumanAnalysisContext(
        segments=[human_by_id[segment.segment_id] for segment in document_segments]
    )
    return document, context


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
    analysis: IntermediateAnalysis | HumanAnalysisContext | None = None,
) -> GenerationRequest:
    return GenerationRequest(
        condition=condition,
        document=document,
        rubric=rubric,
        analysis=analysis,
    )