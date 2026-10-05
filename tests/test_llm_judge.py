from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from reflection_assessment_feedback.llm_judge import (
    AssessmentQualityJudgeOutput,
    FeedbackQualityJudgeOutput,
    build_judge_input,
    run_quality_judge,
)


class FakeRunnable:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.prompt: str | None = None

    def invoke(self, prompt: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        self.prompt = prompt
        return self.result


class FakeChatModel:
    model = "fake-judge-model"
    temperature = 0.0

    def __init__(self, result: dict[str, Any]) -> None:
        self.runnable = FakeRunnable(result)
        self.schema: type | None = None

    def with_structured_output(self, schema: type) -> FakeRunnable:
        self.schema = schema
        return self.runnable


def _packet(task: str) -> dict[str, Any]:
    return {
        "packet_id": "opaque-packet-id",
        "task": task,
        "condition": "G3",
        "model_name": "generation-model-secret",
        "human_protocol": "human-only instruction secret",
        "reference_teacher_feedback": "Synthetic teacher-educator reference feedback.",
        "gold_analysis_summary": {
            "summary_version": "1.0.0",
            "analysis_source": "provisional_gold_candidate_annotations",
            "aggregation_unit": "candidate_annotation",
            "segment_count": 1,
            "candidate_annotation_count": 1,
            "segments_with_multiple_candidates": 0,
            "candidate_annotations_requiring_review": 0,
            "scope_candidate_counts": {"in_scope": 1, "out_of_scope": 0, "unknown": 0},
            "component_band_summaries": {},
        },
        "rubric_version": "1.0.0",
        "reflection": {"segments": [{"segment_id": "s1", "order": 0, "text": "Synthetic reflection."}]},
        "rubric": {"version": "1.0.0", "dimensions": []},
        "output": (
            {"dimensions": [{"dimension_id": "SW", "score": 2, "justification": "Synthetic.", "evidence_segment_ids": ["s1"]}]}
            if task == "assessment_quality"
            else {"strengths": [{"text": "Synthetic strength.", "evidence_segment_ids": ["s1"]}], "weaknesses": [], "suggestions": []}
        ),
    }


def _rating(criterion_id: str) -> dict[str, Any]:
    return {
        "criterion_id": criterion_id,
        "score": 3,
        "unable_to_judge": False,
        "rationale": "Supported by the synthetic packet.",
    }


def test_judge_uses_task_schema_and_prompt_excludes_blinding_metadata() -> None:
    packet = _packet("assessment_quality")
    model = FakeChatModel({
        "ratings": [
            _rating("score_rubric_fit"),
            _rating("evidence_support"),
            _rating("justification_quality"),
        ],
        "overall_comment": "Synthetic overall comment.",
    })

    result = run_quality_judge(
        model=model,
        packet=packet,
        provider="test-provider",
        model_name="fake-judge-model",
    )

    assert model.schema is AssessmentQualityJudgeOutput
    assert result["task"] == "assessment_quality"
    assert result["judge_model"] == "fake-judge-model"
    assert len(result["input_packet_sha256"]) == 64
    assert len(result["teacher_reference_sha256"]) == 64
    assert len(result["gold_analysis_summary_sha256"]) == 64
    assert model.runnable.prompt is not None
    assert "Synthetic reflection." in model.runnable.prompt
    assert "Synthetic teacher-educator reference feedback." in model.runnable.prompt
    assert '"analysis_source": "provisional_gold_candidate_annotations"' in model.runnable.prompt
    assert "generation-model-secret" not in model.runnable.prompt
    assert "human-only instruction secret" not in model.runnable.prompt
    assert "opaque-packet-id" not in model.runnable.prompt
    assert "G3" not in model.runnable.prompt


def test_feedback_judge_validates_spans_against_feedback_text() -> None:
    packet = _packet("feedback_quality")
    model = FakeChatModel({
        "ratings": [_rating("correctness"), _rating("developmental_usefulness")],
        "overall_comment": "Synthetic overall comment.",
        "span_comments": [
            {
                "feedback_component": "strengths",
                "feedback_item_index": 0,
                "start_character": 0,
                "end_character": 9,
                "tags": ["specific"],
                "comment": "Synthetic span comment.",
                "linked_reflection_segment_ids": ["s1"],
            }
        ],
    })

    result = run_quality_judge(
        model=model,
        packet=packet,
        provider="test-provider",
        model_name="fake-judge-model",
    )

    assert model.schema is FeedbackQualityJudgeOutput
    assert len(result["result"]["span_comments"]) == 1
    assert "gold_analysis_summary_sha256" not in result


def test_feedback_judge_rejects_invalid_span_offsets() -> None:
    packet = _packet("feedback_quality")
    model = FakeChatModel({
        "ratings": [_rating("correctness"), _rating("developmental_usefulness")],
        "overall_comment": "Synthetic overall comment.",
        "span_comments": [
            {
                "feedback_component": "strengths",
                "feedback_item_index": 0,
                "start_character": 0,
                "end_character": 99,
                "tags": [],
                "comment": "Synthetic span comment.",
                "linked_reflection_segment_ids": ["s1"],
            }
        ],
    })

    with pytest.raises(ValueError, match="invalid offsets"):
        run_quality_judge(
            model=model,
            packet=packet,
            provider="test-provider",
            model_name="fake-judge-model",
        )


def test_judge_schema_rejects_missing_criteria_and_inconsistent_null_scores() -> None:
    with pytest.raises(ValidationError, match="exactly these criteria"):
        AssessmentQualityJudgeOutput.model_validate({
            "ratings": [_rating("score_rubric_fit"), _rating("evidence_support"), _rating("unexpected")],
            "overall_comment": "Synthetic.",
        })
    with pytest.raises(ValidationError, match="null score"):
        FeedbackQualityJudgeOutput.model_validate({
            "ratings": [
                {**_rating("correctness"), "score": None},
                _rating("developmental_usefulness"),
            ],
            "overall_comment": "Synthetic.",
            "span_comments": [],
        })


def test_judge_packet_contains_only_task_context() -> None:
    payload = build_judge_input(_packet("assessment_quality"))
    assert set(payload) == {
        "reflection", "rubric", "output", "reference_teacher_feedback", "gold_analysis_summary"
    }
    with pytest.raises(ValueError, match="Only assessment-quality"):
        build_judge_input(_packet("feedback_implied_score"))


def test_judge_packet_requires_teacher_reference_feedback() -> None:
    packet = _packet("assessment_quality")
    packet.pop("reference_teacher_feedback")
    with pytest.raises(ValueError, match="teacher reference feedback"):
        build_judge_input(packet)


def test_assessment_judge_packet_requires_gold_summary() -> None:
    packet = _packet("assessment_quality")
    packet.pop("gold_analysis_summary")
    with pytest.raises(ValueError, match="gold-analysis summary"):
        build_judge_input(packet)


def test_feedback_judge_packet_does_not_include_gold_summary() -> None:
    payload = build_judge_input(_packet("feedback_quality"))
    assert "gold_analysis_summary" not in payload