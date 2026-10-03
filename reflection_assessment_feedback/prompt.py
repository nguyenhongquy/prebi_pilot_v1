from __future__ import annotations

import json

from langchain_core.prompts import PromptTemplate

from reflection_assessment_feedback.experiment_config import load_experiment_config
from reflection_assessment_feedback.models import GenerationRequest, HumanAnalysisContext
from reflection_assessment_feedback.prompt_registry import PromptArtifact, resolve_prompt
from reflection_assessment_feedback.rubric import RUBRIC_ARTIFACT

_EXPERIMENT_CONFIG = load_experiment_config()
_GENERATION_CONFIG = _EXPERIMENT_CONFIG["generation"]
PROMPT_ARTIFACT = resolve_prompt(
    _GENERATION_CONFIG["prompt_id"],
    _GENERATION_CONFIG["expected_prompt_version"],
)
G2_SUPPLEMENTAL_ARTIFACT = resolve_prompt(
    _GENERATION_CONFIG["g2_supplemental_prompt_id"],
    _GENERATION_CONFIG["g2_supplemental_prompt_version"],
)
G3_SUPPLEMENTAL_ARTIFACT = resolve_prompt(
    _GENERATION_CONFIG["g3_supplemental_prompt_id"],
    _GENERATION_CONFIG["g3_supplemental_prompt_version"],
)
PROMPT_VERSION = PROMPT_ARTIFACT.version
SYSTEM_INSTRUCTIONS = PROMPT_ARTIFACT.text.rstrip()


def prompt_components_for_request(request: GenerationRequest) -> tuple[PromptArtifact, ...]:
    components = [PROMPT_ARTIFACT]
    if isinstance(request.analysis, HumanAnalysisContext):
        components.append(G3_SUPPLEMENTAL_ARTIFACT)
    elif request.analysis is not None:
        components.append(G2_SUPPLEMENTAL_ARTIFACT)
    return tuple(components)


PROMPT_TEMPLATE = PromptTemplate.from_template(
    f"{SYSTEM_INSTRUCTIONS}\n\n"
    "{supplemental_section}\n\n"
    "Arbeitsgrundlage (JSON):\n{payload}"
)


def build_prompt(request: GenerationRequest) -> str:
    if request.rubric.model_dump(mode="json") != RUBRIC_ARTIFACT.rubric.model_dump(mode="json"):
        raise ValueError("Generation request rubric does not match the configured rubric artifact.")
    payload = {
        "rubric": request.rubric.model_dump(mode="json"),
        "reflection": {
            "document_id": request.document.document_id,
            "segments": [
                segment.model_dump(mode="json") for segment in request.document.segments
            ],
        },
    }
    supplemental_section = ""
    if isinstance(request.analysis, HumanAnalysisContext):
        human_segments = {
            annotation.segment_id: annotation for annotation in request.analysis.segments
        }
        payload["reflection"]["segments"] = [
            {
                **segment.model_dump(mode="json"),
                "human_annotation_candidates": [
                    candidate.model_dump(mode="json")
                    for candidate in human_segments[segment.segment_id].candidates
                ],
            }
            for segment in request.document.segments
        ]
        candidate_count = sum(
            len(segment.candidates) for segment in request.analysis.segments
        )
        reviewed_segment_count = sum(
            any(candidate.requires_review for candidate in segment.candidates)
            for segment in request.analysis.segments
        )
        multiple_candidate_segment_count = sum(
            len(segment.candidates) > 1 for segment in request.analysis.segments
        )
        payload["segment_analysis"] = {
            "source": "human",
            "representation": "candidate_grain_observed_annotations",
            "candidate_annotation_count": candidate_count,
            "reviewed_segment_count": reviewed_segment_count,
            "multiple_candidate_segment_count": multiple_candidate_segment_count,
        }
        supplemental_section = G3_SUPPLEMENTAL_ARTIFACT.text.rstrip()
    elif request.analysis is not None:
        annotations_by_id = {
            annotation.segment_id: annotation for annotation in request.analysis.segments
        }
        payload["reflection"]["segments"] = [
            {
                **segment.model_dump(mode="json"),
                "analysis": {
                    "scope": annotations_by_id[segment.segment_id].scope,
                    "component_bands": {
                        dimension_id: "not_present" if band == "0" else band
                        for dimension_id, band in annotations_by_id[
                            segment.segment_id
                        ].component_bands.items()
                    },
                },
            }
            for segment in request.document.segments
        ]
        payload["segment_analysis"] = {
            "source": request.analysis.source,
            "summary": request.analysis.model_dump(mode="json")["summary"],
        }
        supplemental_section = G2_SUPPLEMENTAL_ARTIFACT.text.rstrip()

    return PROMPT_TEMPLATE.format(
        supplemental_section=supplemental_section,
        payload=json.dumps(payload, ensure_ascii=False, indent=2),
    )