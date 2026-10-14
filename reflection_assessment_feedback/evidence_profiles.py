from __future__ import annotations

from typing import Any

from reflection_assessment_feedback.models import (
    DIMENSION_IDS,
    GenerationRequest,
    HumanSegmentAnalysis,
)

EXTERNAL_BANDS = {"0": None, "1": "A", "2": "B", "3": "C", "N": "NA"}


def build_evidence_profiles(request: GenerationRequest) -> dict[str, Any] | None:
    """Describe assignments without collapsing candidates or calculating performance."""
    if request.analysis is None:
        return None
    annotations = {item.segment_id: item for item in request.analysis.segments}
    dimensions = []
    for dimension_id in DIMENSION_IDS:
        passages = []
        assigned_ids: set[str] = set()
        band_ids: dict[str, set[str]] = {band: set() for band in ("A", "B", "C")}
        candidate_counts = {band: 0 for band in band_ids}
        for segment in request.document.segments:
            annotation = annotations[segment.segment_id]
            candidates = []
            if isinstance(annotation, HumanSegmentAnalysis):
                for candidate in annotation.candidates:
                    known = candidate.scope_known and candidate.scope == "in_scope" and candidate.known_component_bands[dimension_id]
                    raw = candidate.component_bands[dimension_id]
                    status = (
                        "out_of_scope" if candidate.scope_known and candidate.scope == "out_of_scope"
                        else "unknown" if not known or raw not in {"0", "1", "2", "3"}
                        else "not_assigned" if raw == "0" else "assigned"
                    )
                    candidates.append({
                        "candidate_id": candidate.candidate_id,
                        "status": status,
                        "label": EXTERNAL_BANDS[raw] if status == "assigned" and raw is not None else "NA" if status == "out_of_scope" else None,
                        "scope": candidate.scope,
                        "scope_known": candidate.scope_known,
                        "known": candidate.known_component_bands[dimension_id],
                        "raw_scope_value": candidate.raw_scope_value,
                        "raw_component_value": candidate.raw_component_values[dimension_id],
                        "requires_review": candidate.requires_review,
                    })
            else:
                raw = annotation.component_bands[dimension_id]
                candidates.append({
                    "candidate_id": None,
                    "status": "out_of_scope" if annotation.scope == "out_of_scope" else "not_assigned" if raw == "0" else "assigned",
                    "label": EXTERNAL_BANDS[raw],
                    "scope": annotation.scope,
                    "known": True,
                    "requires_review": False,
                })
            for candidate in candidates:
                if candidate["status"] == "assigned":
                    label = candidate["label"]
                    assigned_ids.add(segment.segment_id)
                    band_ids[label].add(segment.segment_id)
                    candidate_counts[label] += 1
            signatures = {(item["status"], item["label"]) for item in candidates}
            passages.append({
                "segment_id": segment.segment_id,
                "text": segment.text,
                "candidates": candidates,
                "conflicting_annotations": len(signatures) > 1,
                "uncertain": len(signatures) > 1 or any(
                    item["requires_review"] or item["status"] == "unknown" for item in candidates
                ),
            })
        dimensions.append({
            "dimension_id": dimension_id,
            "distinct_assigned_segment_count": len(assigned_ids),
            "distinct_segment_band_counts": {band: len(ids) for band, ids in band_ids.items()},
            "assigned_candidate_annotation_count": sum(candidate_counts.values()),
            "candidate_band_counts": candidate_counts,
            "passages": passages,
        })
    return {
        "source": request.analysis.source,
        "counting_note": "Distinct segments and candidate annotations are separate units. Conflicted segments may appear in more than one band count. Counts are descriptive, not document scores.",
        "dimensions": dimensions,
    }
