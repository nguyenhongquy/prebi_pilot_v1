from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from langsmith import Client, tracing_context
from pydantic import BaseModel, ConfigDict, Field, model_validator

from reflection_assessment_feedback.experiment_config import load_experiment_config
from reflection_assessment_feedback.prompt_registry import render_prompt, resolve_prompt

LANGSMITH_EU_ENDPOINT = "https://eu.api.smith.langchain.com"
JudgeTask = Literal["assessment_quality", "feedback_quality"]


class StrictJudgeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JudgeCriterionRating(StrictJudgeModel):
    criterion_id: str = Field(min_length=1)
    score: int | None = Field(default=None, ge=1, le=3)
    unable_to_judge: bool
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_judgability(self) -> JudgeCriterionRating:
        if self.unable_to_judge != (self.score is None):
            raise ValueError("Unable-to-judge ratings must have a null score; other ratings need a score.")
        return self


class AssessmentQualityJudgeOutput(StrictJudgeModel):
    ratings: list[JudgeCriterionRating] = Field(min_length=3, max_length=3)
    overall_comment: str

    @model_validator(mode="after")
    def validate_criteria(self) -> AssessmentQualityJudgeOutput:
        _validate_criterion_ids(
            self.ratings,
            {"score_rubric_fit", "evidence_support", "justification_quality"},
        )
        return self


class FeedbackSpanComment(StrictJudgeModel):
    feedback_component: Literal["strengths", "weaknesses", "suggestions"]
    feedback_item_index: int = Field(ge=0)
    start_character: int = Field(ge=0)
    end_character: int = Field(ge=1)
    tags: list[str]
    comment: str = Field(min_length=1)
    linked_reflection_segment_ids: list[str]


class FeedbackQualityJudgeOutput(StrictJudgeModel):
    ratings: list[JudgeCriterionRating] = Field(min_length=2, max_length=2)
    overall_comment: str
    span_comments: list[FeedbackSpanComment]

    @model_validator(mode="after")
    def validate_criteria(self) -> FeedbackQualityJudgeOutput:
        _validate_criterion_ids(self.ratings, {"correctness", "developmental_usefulness"})
        return self


JUDGE_OUTPUT_MODELS: dict[str, type[StrictJudgeModel]] = {
    "assessment_quality": AssessmentQualityJudgeOutput,
    "feedback_quality": FeedbackQualityJudgeOutput,
}


def _validate_criterion_ids(ratings: list[JudgeCriterionRating], expected: set[str]) -> None:
    actual = [rating.criterion_id for rating in ratings]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(f"Ratings must contain exactly these criteria: {sorted(expected)}.")


def _canonical_sha256(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_judge_input(packet: dict[str, Any]) -> dict[str, Any]:
    task = packet.get("task")
    if task not in JUDGE_OUTPUT_MODELS:
        raise ValueError("Only assessment-quality and feedback-quality packets can be judged.")
    required = {"reflection", "rubric", "output", "reference_teacher_feedback"}
    if task == "assessment_quality":
        required.add("gold_analysis_summary")
    if not required.issubset(packet):
        raise ValueError(
            "Judge packet is missing reflection, rubric, generated output, teacher reference feedback, "
            "or the assessment gold-analysis summary."
        )
    reference_feedback = packet["reference_teacher_feedback"]
    if not isinstance(reference_feedback, str) or not reference_feedback.strip():
        raise ValueError("Teacher reference feedback must be a non-empty string.")
    judge_input = {
        name: packet[name]
        for name in ("reflection", "rubric", "output", "reference_teacher_feedback")
    }
    if task == "assessment_quality":
        summary = packet["gold_analysis_summary"]
        if not isinstance(summary, dict) or summary.get("analysis_source") != "provisional_gold_candidate_annotations":
            raise ValueError("Assessment judge requires a provisional gold-analysis summary.")
        judge_input["gold_analysis_summary"] = summary
    return judge_input


def judge_input_sha256(packet: dict[str, Any]) -> str:
    return _canonical_sha256(build_judge_input(packet))


def _validate_span_comments(
    output: FeedbackQualityJudgeOutput,
    judge_input: dict[str, Any],
) -> None:
    feedback = judge_input["output"]
    segment_ids = {
        segment["segment_id"]
        for segment in judge_input["reflection"].get("segments", [])
        if "segment_id" in segment
    }
    for comment in output.span_comments:
        items = feedback.get(comment.feedback_component, [])
        if comment.feedback_item_index >= len(items):
            raise ValueError("A span comment references a missing feedback item.")
        text = items[comment.feedback_item_index].get("text", "")
        if comment.start_character >= comment.end_character or comment.end_character > len(text):
            raise ValueError("A span comment has invalid offsets for its feedback item.")
        if not set(comment.linked_reflection_segment_ids).issubset(segment_ids):
            raise ValueError("A span comment references an unknown reflection segment.")


def run_quality_judge(
    *,
    model: Any,
    packet: dict[str, Any],
    provider: str,
    model_name: str,
    enable_langsmith_tracing: bool = False,
) -> dict[str, Any]:
    task = packet.get("task")
    if task not in JUDGE_OUTPUT_MODELS:
        raise ValueError("Only assessment-quality and feedback-quality packets can be judged.")
    if not provider.strip() or not model_name.strip():
        raise ValueError("Judge provider and model name must be configured.")

    prompt_config = load_experiment_config()["prompts"]
    prompt_artifact = resolve_prompt(
        prompt_config[f"{task}_llm_id"],
        prompt_config[f"{task}_llm_version"],
    )
    if prompt_artifact.task != task or prompt_artifact.modality != "llm_prompt":
        raise ValueError("Configured judge prompt is not registered for this task.")
    if prompt_artifact.status != "development_draft":
        raise ValueError("This runner is restricted to development-draft judge prompts.")

    judge_input = build_judge_input(packet)
    packet_json = json.dumps(judge_input, ensure_ascii=False, sort_keys=True, indent=2)
    rendered_prompt, rendered_prompt_sha256 = render_prompt(
        prompt_artifact,
        packet_json=packet_json,
    )
    packet_sha256 = _canonical_sha256(judge_input)
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
            raise RuntimeError("LLM judging requires the LangSmith EU endpoint.")
        tracing_client = Client(api_url=api_url, api_key=api_key)
        tracing_client.read_project(project_name=project_name)

    runnable = model.with_structured_output(JUDGE_OUTPUT_MODELS[task])
    trace_metadata = {
        "task": task,
        "run_id": input_run_id,
        "prompt_id": prompt_artifact.artifact_id,
        "prompt_version": prompt_artifact.version,
        "prompt_artifact_sha256": prompt_artifact.sha256,
        "rendered_prompt_sha256": rendered_prompt_sha256,
    }
    with tracing_context(
        enabled=enable_langsmith_tracing,
        project_name=project_name,
        client=tracing_client,
        tags=["llm-judge", task, "development"],
        metadata=trace_metadata,
    ):
        raw_output = runnable.invoke(
            rendered_prompt,
            config={
                "run_name": f"{task}-llm-judge",
                "tags": ["llm-judge", task, "development"],
                "metadata": trace_metadata,
            },
        )
    if tracing_client is not None:
        tracing_client.flush()

    output_model = JUDGE_OUTPUT_MODELS[task]
    validated_output = (
        raw_output
        if isinstance(raw_output, output_model)
        else output_model.model_validate(raw_output)
    )
    if isinstance(validated_output, FeedbackQualityJudgeOutput):
        _validate_span_comments(validated_output, judge_input)

    result = {
        "schema_version": "1.0.0",
        "record_type": "llm_judge_quality_rating",
        "run_id": input_run_id,
        "packet_id": packet.get("packet_id"),
        "task": task,
        "status": "development_judged",
        "evaluator_type": "llm_judge",
        "judge_provider": provider,
        "judge_model": model_name,
        "judge_parameters": {
            name: value
            for name in ("top_p", "max_tokens", "seed")
            if (value := getattr(model, name, None)) is not None
        },
        "prompt_id": prompt_artifact.artifact_id,
        "prompt_version": prompt_artifact.version,
        "prompt_artifact_sha256": prompt_artifact.sha256,
        "rendered_prompt_sha256": rendered_prompt_sha256,
        "input_packet_sha256": packet_sha256,
        "teacher_reference_sha256": hashlib.sha256(
            judge_input["reference_teacher_feedback"].encode("utf-8")
        ).hexdigest(),
        "rubric_version": packet.get("rubric_version"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result": validated_output.model_dump(mode="json"),
    }
    if task == "assessment_quality":
        result["gold_analysis_summary_sha256"] = _canonical_sha256(
            judge_input["gold_analysis_summary"]
        )
    return result