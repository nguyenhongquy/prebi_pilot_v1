from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from langsmith import Client, tracing_context
from pydantic import BaseModel, ConfigDict, Field, model_validator

from reflection_assessment_feedback.experiment_config import (
    experiment_config_sha256,
    load_experiment_config,
)
from reflection_assessment_feedback.models import DIMENSION_IDS, ReflectionDocument, Rubric
from reflection_assessment_feedback.prompt_registry import render_prompt, resolve_prompt
from reflection_assessment_feedback.rubric import RUBRIC_ARTIFACT

LANGSMITH_EU_ENDPOINT = "https://eu.api.smith.langchain.com"


class StrictScoreModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FeedbackScoreEvidence(StrictScoreModel):
    start_character: int | None = Field(default=None, ge=0)
    end_character: int | None = Field(default=None, ge=1)
    quote: str = Field(min_length=1)


class FeedbackImpliedDimensionScore(StrictScoreModel):
    dimension_id: Literal["SW", "UA", "HA"]
    status: Literal["inferred_from_feedback", "not_inferable"]
    score: float | None = Field(default=None, ge=0, le=3, multiple_of=0.1)
    confidence: Literal["low", "medium", "high", "not_inferable"]
    rationale: str = Field(min_length=1)
    feedback_evidence: list[FeedbackScoreEvidence]

    @model_validator(mode="after")
    def validate_inferability_fields(self) -> FeedbackImpliedDimensionScore:
        if self.status == "not_inferable":
            if self.score is not None or self.confidence != "not_inferable":
                raise ValueError("Not-inferable scores require null score and confidence.")
            if self.feedback_evidence:
                raise ValueError("Not-inferable scores must not cite evidence as score support.")
        elif self.score is None or self.confidence == "not_inferable":
            raise ValueError("Inferred scores require a score and low/medium/high confidence.")
        elif not self.feedback_evidence:
            raise ValueError("Inferred scores require feedback evidence spans.")
        return self


class FeedbackImpliedScoreOutput(StrictScoreModel):
    scores: list[FeedbackImpliedDimensionScore] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_dimensions(self) -> FeedbackImpliedScoreOutput:
        dimensions = [item.dimension_id for item in self.scores]
        if len(dimensions) != len(set(dimensions)) or set(dimensions) != set(DIMENSION_IDS):
            raise ValueError("Output must contain exactly one score decision for SW, UA, and HA.")
        return self


def build_feedback_score_packet(
    *,
    packet_id: str,
    document: ReflectionDocument,
    rubric: Rubric,
    human_feedback: str,
) -> dict[str, Any]:
    if not human_feedback.strip():
        raise ValueError("Human feedback must not be empty.")
    return {
        "packet_id": packet_id,
        "reflection": {
            "segments": [
                {
                    "segment_id": segment.segment_id,
                    "order": segment.order,
                    "text": segment.text,
                }
                for segment in document.segments
            ]
        },
        "rubric": rubric.model_dump(mode="json"),
        "human_feedback": human_feedback,
    }


def feedback_score_packet_sha256(packet: dict[str, Any]) -> str:
    packet_json = json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packet_json.encode("utf-8")).hexdigest()


def _validate_evidence_offsets(
    output: FeedbackImpliedScoreOutput,
    feedback_text: str,
) -> list[list[dict[str, Any]]]:
    all_statuses = []
    for score in output.scores:
        score_statuses = []
        for evidence in score.feedback_evidence:
            status: dict[str, Any] = {
                "status": "verified",
                "suggested_start_character": None,
                "suggested_end_character": None,
            }
            if evidence.start_character is None or evidence.end_character is None:
                status["status"] = "offsets_not_provided"
            elif evidence.start_character >= evidence.end_character:
                status["status"] = "invalid_offsets"
            elif evidence.end_character > len(feedback_text):
                status["status"] = "offsets_out_of_range"
            elif feedback_text[evidence.start_character:evidence.end_character] != evidence.quote:
                matches = []
                search_from = 0
                while True:
                    match_at = feedback_text.find(evidence.quote, search_from)
                    if match_at < 0:
                        break
                    matches.append(match_at)
                    search_from = match_at + 1
                if len(matches) == 1:
                    status["status"] = "offset_mismatch_quote_found_uniquely"
                    status["suggested_start_character"] = matches[0]
                    status["suggested_end_character"] = matches[0] + len(evidence.quote)
                elif matches:
                    status["status"] = "quote_found_multiple_times"
                else:
                    status["status"] = "quote_not_found"
            score_statuses.append(status)
        all_statuses.append(score_statuses)
    return all_statuses


def infer_feedback_implied_score(
    *,
    model: Any,
    document: ReflectionDocument,
    rubric: Rubric,
    human_feedback: str,
    human_feedback_id: str,
    document_id: str,
    enable_langsmith_tracing: bool,
    packet_id: str | None = None,
) -> dict[str, Any]:
    config = load_experiment_config()
    prompt_config = config["prompts"]
    prompt_artifact = resolve_prompt(
        prompt_config["feedback_implied_score_llm_id"],
        prompt_config["feedback_implied_score_llm_version"],
    )
    if prompt_artifact.status != "development_draft":
        raise ValueError("This pilot runner is restricted to a development-draft scoring prompt.")

    packet_id = packet_id or str(uuid4())
    packet = build_feedback_score_packet(
        packet_id=packet_id,
        document=document,
        rubric=rubric,
        human_feedback=human_feedback,
    )
    packet_json = json.dumps(packet, ensure_ascii=False, indent=2)
    rendered_prompt, rendered_prompt_sha256 = render_prompt(
        prompt_artifact,
        packet_json=packet_json,
    )
    packet_sha256 = feedback_score_packet_sha256(packet)
    input_run_id = str(uuid4())

    tracing_client = None
    project_name = None
    if enable_langsmith_tracing:
        api_key = os.environ.get("LANGSMITH_API_KEY")
        project_name = os.environ.get("LANGSMITH_PROJECT")
        api_url = os.environ.get("LANGSMITH_ENDPOINT", LANGSMITH_EU_ENDPOINT).rstrip("/")
        if not api_key or not project_name:
            raise RuntimeError("LangSmith tracing requires an API key and project name.")
        if api_url != LANGSMITH_EU_ENDPOINT:
            raise RuntimeError("Feedback-implied scoring requires the LangSmith EU endpoint.")
        tracing_client = Client(api_url=api_url, api_key=api_key)
        tracing_client.read_project(project_name=project_name)

    runnable = model.with_structured_output(FeedbackImpliedScoreOutput)
    with tracing_context(
        enabled=enable_langsmith_tracing,
        project_name=project_name,
        client=tracing_client,
        tags=["feedback-implied-score", "development"],
        metadata={
            "task": "feedback_implied_score",
            "run_id": input_run_id,
            "packet_id": packet_id,
            "prompt_id": prompt_artifact.artifact_id,
            "prompt_version": prompt_artifact.version,
            "prompt_artifact_sha256": prompt_artifact.sha256,
            "rendered_prompt_sha256": rendered_prompt_sha256,
        },
    ):
        raw_output = runnable.invoke(
            rendered_prompt,
            config={
                "run_name": "feedback-implied-score-inference",
                "tags": ["feedback-implied-score", "development"],
                "metadata": {
                    "run_id": input_run_id,
                    "packet_id": packet_id,
                    "prompt_id": prompt_artifact.artifact_id,
                    "prompt_version": prompt_artifact.version,
                },
            },
        )
    if tracing_client is not None:
        tracing_client.flush()

    output = (
        raw_output
        if isinstance(raw_output, FeedbackImpliedScoreOutput)
        else FeedbackImpliedScoreOutput.model_validate(raw_output)
    )
    evidence_validation = _validate_evidence_offsets(output, human_feedback)
    model_name = str(
        getattr(model, "model", None)
        or getattr(model, "model_name", None)
        or type(model).__name__
    )
    generation_parameters = {
        name: value
        for name in ("temperature", "top_p", "top_k", "max_output_tokens", "seed")
        if (value := getattr(model, name, None)) is not None
    }
    created_at = datetime.now(timezone.utc).isoformat()
    score_records = []
    for score, evidence_statuses in zip(output.scores, evidence_validation, strict=True):
        score_data = score.model_dump(mode="json")
        score_data["feedback_evidence"] = [
            {**evidence.model_dump(mode="json"), **status}
            for evidence, status in zip(score.feedback_evidence, evidence_statuses, strict=True)
        ]
        score_records.append({
            "feedback_score_id": str(uuid4()),
            "run_id": input_run_id,
            "document_id": document_id,
            "human_feedback_id": human_feedback_id,
            **score_data,
            "evaluator_type": "llm_inference",
            "evaluator_id": model_name,
            "human_protocol_id": None,
            "human_protocol_version": None,
            "llm_prompt_id": prompt_artifact.artifact_id,
            "llm_prompt_version": prompt_artifact.version,
            "rendered_prompt_sha256": rendered_prompt_sha256,
            "created_at": created_at,
        })
    return {
        "schema_version": "1.0.0",
        "record_type": "feedback_implied_score_inference",
        "run_id": input_run_id,
        "packet_id": packet_id,
        "document_id": document_id,
        "human_feedback_id": human_feedback_id,
        "status": "development_inferred",
        "evaluator_type": "llm_inference",
        "model_provider": config["generation"]["provider"],
        "model_name": model_name,
        "generation_parameters": generation_parameters,
        "prompt_id": prompt_artifact.artifact_id,
        "prompt_version": prompt_artifact.version,
        "prompt_artifact_sha256": prompt_artifact.sha256,
        "rendered_prompt_sha256": rendered_prompt_sha256,
        "rubric_id": rubric.rubric_id,
        "rubric_version": rubric.version,
        "input_packet_sha256": packet_sha256,
        "experiment_config_sha256": experiment_config_sha256(),
        "created_at": created_at,
        "rubric_artifact_sha256": RUBRIC_ARTIFACT.sha256,
        "scores": score_records,
    }
