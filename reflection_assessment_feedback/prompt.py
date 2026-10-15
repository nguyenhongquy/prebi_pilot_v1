from __future__ import annotations

import json

from langchain_core.prompts import PromptTemplate

from reflection_assessment_feedback.experiment_config import load_experiment_config
from reflection_assessment_feedback.models import GenerationRequest
from reflection_assessment_feedback.prompt_registry import PromptArtifact, resolve_prompt
from reflection_assessment_feedback.rubric import RUBRIC_ARTIFACT
from reflection_assessment_feedback.evidence_profiles import build_evidence_profiles

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


def prompt_components_for_request(
    request: GenerationRequest, *, g3_supplemental_version: str | None = None,
) -> tuple[PromptArtifact, ...]:
    if g3_supplemental_version is not None and request.condition != "G3":
        raise ValueError("G3 prompt override requires a G3 request.")
    components = [PROMPT_ARTIFACT]
    if request.analysis is not None and request.analysis.source == "human":
        components.append(
            resolve_prompt(G3_SUPPLEMENTAL_ARTIFACT.artifact_id, g3_supplemental_version)
            if g3_supplemental_version is not None else G3_SUPPLEMENTAL_ARTIFACT
        )
    elif request.analysis is not None:
        components.append(G2_SUPPLEMENTAL_ARTIFACT)
    return tuple(components)


PROMPT_TEMPLATE = PromptTemplate.from_template(
    f"{SYSTEM_INSTRUCTIONS}\n\n"
    "{supplemental_section}\n\n"
    "Arbeitsgrundlage (JSON):\n{payload}"
)


def build_prompt(request: GenerationRequest, *, g3_supplemental_version: str | None = None) -> str:
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
    components = prompt_components_for_request(request, g3_supplemental_version=g3_supplemental_version)
    supplemental_section = ""
    profile = build_evidence_profiles(request)
    if profile is not None and request.analysis is not None:
        if g3_supplemental_version == "2.2.0":
            payload["segment_dimension_assignments"] = [
                {
                    "segment_id": segment.segment_id,
                    "dimension_ids": [
                        dimension["dimension_id"] for dimension in profile["dimensions"]
                        if any(
                            candidate["status"] == "assigned"
                            for passage in dimension["passages"]
                            if passage["segment_id"] == segment.segment_id
                            for candidate in passage["candidates"]
                        )
                    ],
                }
                for segment in request.document.segments
            ]
        else:
            payload["evidence_profile"] = profile
        supplemental_section = components[-1].text.rstrip()

    return PROMPT_TEMPLATE.format(
        supplemental_section=supplemental_section,
        payload=json.dumps(payload, ensure_ascii=False, indent=2),
    )