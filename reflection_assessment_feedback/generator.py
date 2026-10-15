from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langsmith import Client, traceable, tracing_context
from pydantic import BaseModel
from langsmith.utils import tracing_is_enabled

from reflection_assessment_feedback.models import (
    GenerationOutput,
    DocumentGenerationOutput,
    GenerationRequest,
    PIPELINE_VERSION,
    ReflectionDocument,
)
from reflection_assessment_feedback.prompt import (
    PROMPT_ARTIFACT,
    PROMPT_VERSION,
    build_prompt,
    prompt_components_for_request,
)
from reflection_assessment_feedback.rubric import RUBRIC_ARTIFACT

LANGSMITH_EU_ENDPOINT = "https://eu.api.smith.langchain.com"


class StructuredRunnable(Protocol):
    def invoke(self, input: str, config: RunnableConfig | None = None) -> Any: ...


class StructuredChatModel(Protocol):
    def with_structured_output(
        self,
        schema: type[BaseModel],
    ) -> StructuredRunnable: ...


@dataclass(frozen=True)
class GenerationRun:
    run_id: str
    condition: str
    repetition: int
    model_name: str
    generation_parameters: dict[str, Any]
    prompt_version: str
    prompt_artifact_sha256: str
    rendered_prompt_sha256: str
    prompt_components: list[dict[str, str]]
    rubric_artifact_sha256: str
    output: DocumentGenerationOutput
    feedback_letter: str
    pipeline_version: str = PIPELINE_VERSION
    rubric_version: str = "1.2.0"


def validate_output(request: GenerationRequest, output: GenerationOutput | DocumentGenerationOutput) -> None:
    if request.rubric.version == "1.2.0":
        output = DocumentGenerationOutput.model_validate(output.model_dump(mode="json"))
    dimensions = {dimension.dimension_id: dimension for dimension in request.rubric.dimensions}
    assessments = output.assessment.component_assessments
    assessment_ids = [assessment.dimension_id for assessment in assessments]
    if len(assessment_ids) != len(set(assessment_ids)) or set(assessment_ids) != set(dimensions):
        raise ValueError("Output must contain exactly one assessment for every rubric dimension.")

    segment_ids = {segment.segment_id for segment in request.document.segments}
    if isinstance(output, DocumentGenerationOutput):
        for synthesis in output.evidence_synthesis:
            for field in ("demonstrated_depth", "development", "integrative_coherence", "limitations", "uncertainty"):
                for claim in getattr(synthesis, field):
                    _validate_evidence_ids(claim.evidence_refs, segment_ids)
    for assessment in assessments:
        dimension = dimensions[assessment.dimension_id]
        rubric_band_values = [int(band) for band in dimension.bands]
        if not min(rubric_band_values) <= assessment.band <= max(rubric_band_values):
            raise ValueError("An assessment score is outside the rubric band range.")
        _validate_evidence_ids(assessment.evidence_segment_ids, segment_ids)

    feedback_groups = (
        output.feedback.strengths,
        output.feedback.weaknesses,
        output.feedback.suggestions,
    )
    for group in feedback_groups:
        for point in group:
            _validate_evidence_ids(point.evidence_segment_ids, segment_ids)


def _validate_evidence_ids(evidence_ids: list[str], segment_ids: set[str]) -> None:
    unknown = set(evidence_ids) - segment_ids
    if unknown:
        raise ValueError("Output cites a segment ID that is not present in the input.")


def output_for_review(output: GenerationOutput | DocumentGenerationOutput, document: ReflectionDocument) -> dict[str, Any]:
    segment_orders = {segment.segment_id: segment.order for segment in document.segments}
    if isinstance(output, DocumentGenerationOutput):
        for synthesis in output.evidence_synthesis:
            for field in ("demonstrated_depth", "development", "integrative_coherence", "limitations", "uncertainty"):
                for claim in getattr(synthesis, field):
                    _validate_evidence_ids(claim.evidence_refs, set(segment_orders))
    result = output.model_dump(mode="json")
    for assessment in result["assessment"]["component_assessments"]:
        _validate_evidence_ids(assessment["evidence_segment_ids"], set(segment_orders))
        assessment["evidence_segment_orders"] = [
            segment_orders[segment_id] for segment_id in assessment["evidence_segment_ids"]
        ]
    return result


def render_feedback_letter(output: GenerationOutput | DocumentGenerationOutput) -> str:
    sections = (
        ("Stärken", output.feedback.strengths),
        ("Entwicklungsbedarfe", output.feedback.weaknesses),
        ("Vorschläge für nächste Schritte", output.feedback.suggestions),
    )
    rendered = [
        f"## {heading}\n" + "\n".join(f"- {point.message}" for point in points)
        for heading, points in sections
        if points
    ]
    if not rendered:
        raise ValueError("At least one feedback component is required to render a letter.")
    return "\n\n".join(rendered)


class RubricGenerationRunner:
    def __init__(
        self,
        model: StructuredChatModel,
        *,
        enable_langsmith_tracing: bool = False,
        g3_supplemental_version: str | None = None,
    ) -> None:
        self._model = model
        self._enable_langsmith_tracing = enable_langsmith_tracing
        self._g3_supplemental_version = g3_supplemental_version

    def generate(self, request: GenerationRequest, *, repetition: int = 1) -> GenerationRun:
        prompt_components_for_request(request, g3_supplemental_version=self._g3_supplemental_version)
        if repetition < 1:
            raise ValueError("repetition must be a positive integer.")
        environment_tracing = any(
            os.environ.get(name, "").strip().lower() in {"true", "1", "yes"}
            for name in (
                "LANGSMITH_TRACING",
                "LANGCHAIN_TRACING_V2",
                "LANGCHAIN_TRACING",
            )
        )
        if (
            not self._enable_langsmith_tracing
            and (tracing_is_enabled() or environment_tracing)
        ):
            raise RuntimeError(
                "LangSmith tracing is active in the current environment or context. "
                "Disable tracing or explicitly opt in after confirming data approval."
            )

        project_name = None
        client = None
        if self._enable_langsmith_tracing:
            api_key = os.environ.get("LANGSMITH_API_KEY")
            project_name = os.environ.get("LANGSMITH_PROJECT")
            api_url = os.environ.get("LANGSMITH_ENDPOINT", LANGSMITH_EU_ENDPOINT).rstrip("/")
            if not api_key:
                raise RuntimeError(
                    "LangSmith tracing requires LANGSMITH_API_KEY when enabled."
                )
            if not project_name:
                raise RuntimeError(
                    "LangSmith tracing requires LANGSMITH_PROJECT when enabled."
                )
            if api_url != LANGSMITH_EU_ENDPOINT:
                raise RuntimeError(
                    "This experiment requires LANGSMITH_ENDPOINT to be "
                    f"{LANGSMITH_EU_ENDPOINT}."
                )
            client = Client(api_url=api_url, api_key=api_key)
            client.read_project(project_name=project_name)

        with tracing_context(
            enabled=self._enable_langsmith_tracing,
            project_name=project_name,
            client=client,
            tags=["direct-generation", request.condition],
            metadata={"condition": request.condition, "repetition": repetition},
        ):
            result = self._generate_traced(request, repetition=repetition)
        if client is not None:
            client.flush()
        return result

    @traceable(name="rubric_generation", run_type="chain")
    def _generate_traced(
        self,
        request: GenerationRequest,
        *,
        repetition: int,
    ) -> GenerationRun:
        run_id = str(uuid4())
        model_name = str(
            getattr(self._model, "model", None)
            or getattr(self._model, "model_name", None)
            or type(self._model).__name__
        )
        generation_parameters = {
            name: value
            for name in ("temperature", "top_p", "top_k", "max_output_tokens", "seed")
            if (value := getattr(self._model, name, None)) is not None
        }
        runnable = self._model.with_structured_output(DocumentGenerationOutput)
        prompt_text = build_prompt(request, g3_supplemental_version=self._g3_supplemental_version)
        rendered_prompt_sha256 = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        prompt_components = [
            {
                "artifact_id": artifact.artifact_id,
                "version": artifact.version,
                "sha256": artifact.sha256,
            }
            for artifact in prompt_components_for_request(
                request, g3_supplemental_version=self._g3_supplemental_version,
            )
        ]
        config: RunnableConfig = {
            "run_name": f"rubric-generation-{request.condition.lower()}",
            "tags": ["direct-generation", request.condition],
            "metadata": {
                "condition": request.condition,
                "repetition": repetition,
                "run_id": run_id,
                "model": model_name,
                "generation_parameters": generation_parameters,
                "rubric_id": request.rubric.rubric_id,
                "rubric_version": request.rubric.version,
                "pipeline_version": PIPELINE_VERSION,
                "prompt_version": PROMPT_VERSION,
                "prompt_artifact_sha256": PROMPT_ARTIFACT.sha256,
                "rendered_prompt_sha256": rendered_prompt_sha256,
                "prompt_components": prompt_components,
                "rubric_artifact_sha256": RUBRIC_ARTIFACT.sha256,
            },
        }
        if not self._enable_langsmith_tracing:
            config["callbacks"] = []
        result = runnable.invoke(
            prompt_text,
            config=config,
        )
        output = (
            result
            if isinstance(result, DocumentGenerationOutput)
            else DocumentGenerationOutput.model_validate(
                result.model_dump(mode="json") if isinstance(result, BaseModel) else result
            )
        )
        validate_output(request, output)
        return GenerationRun(
            run_id=run_id,
            condition=request.condition,
            repetition=repetition,
            model_name=model_name,
            rubric_version=request.rubric.version,
            generation_parameters=generation_parameters,
            prompt_version=PROMPT_VERSION,
            prompt_artifact_sha256=PROMPT_ARTIFACT.sha256,
            rendered_prompt_sha256=rendered_prompt_sha256,
            prompt_components=prompt_components,
            rubric_artifact_sha256=RUBRIC_ARTIFACT.sha256,
            output=output,
            feedback_letter=render_feedback_letter(output),
        )