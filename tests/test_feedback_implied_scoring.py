from __future__ import annotations

from typing import Any

import pytest
from jsonschema import Draft202012Validator

from reflection_assessment_feedback import EXPERT_RUBRIC, ReflectionDocument, ReflectionSegment
from reflection_assessment_feedback.feedback_implied_scoring import (
    FeedbackImpliedScoreOutput,
    build_feedback_score_packet,
    infer_feedback_implied_score,
)


class FakeRunnable:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.prompt: str | None = None

    def invoke(self, prompt: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        self.prompt = prompt
        return self.result


class FakeChatModel:
    model = "fake-score-model"
    temperature = 0.0

    def __init__(self, result: dict[str, Any]) -> None:
        self.runnable = FakeRunnable(result)

    def with_structured_output(self, schema: type) -> FakeRunnable:
        assert schema is FeedbackImpliedScoreOutput
        return self.runnable


def _valid_response(feedback: str) -> dict[str, Any]:
    start = feedback.index("2.3")
    end = start + len("2.3")
    return {
        "scores": [
            {
                "dimension_id": "SW",
                "status": "inferred_from_feedback",
                "score": 2.3,
                "confidence": "medium",
                "rationale": "Feedback explicitly signals this band-level judgment.",
                "feedback_evidence": [{"start_character": start, "end_character": end, "quote": "2.3"}],
            },
            {
                "dimension_id": "UA",
                "status": "not_inferable",
                "score": None,
                "confidence": "not_inferable",
                "rationale": "No dimension-specific score is communicated.",
                "feedback_evidence": [],
            },
            {
                "dimension_id": "HA",
                "status": "not_inferable",
                "score": None,
                "confidence": "not_inferable",
                "rationale": "Feedback does not support a defensible score.",
                "feedback_evidence": [],
            },
        ]
    }


def test_build_packet_excludes_document_id_but_contains_feedback_context() -> None:
    document = ReflectionDocument(
        document_id="private-document-id",
        segments=[ReflectionSegment(segment_id="s1", text="Reflection passage", order=0)],
    )
    packet = build_feedback_score_packet(
        packet_id="opaque-packet",
        document=document,
        rubric=EXPERT_RUBRIC,
        human_feedback="Feedback 2.3",
    )

    assert packet["packet_id"] == "opaque-packet"
    assert "document_id" not in packet
    assert packet["human_feedback"] == "Feedback 2.3"
    assert packet["reflection"]["segments"][0]["segment_id"] == "s1"
    assert "rubric" in packet


def test_inference_emits_provenance_valid_scores_and_exact_evidence() -> None:
    feedback = "Feedback 2.3"
    model = FakeChatModel(_valid_response(feedback))
    document = ReflectionDocument(
        document_id="doc-1",
        segments=[ReflectionSegment(segment_id="s1", text="Reflection passage", order=0)],
    )

    result = infer_feedback_implied_score(
        model=model,
        document=document,
        rubric=EXPERT_RUBRIC,
        human_feedback=feedback,
        human_feedback_id="feedback-1",
        document_id="doc-1",
        enable_langsmith_tracing=False,
    )

    assert result["status"] == "development_inferred"
    assert result["prompt_id"] == "feedback_implied_score_llm_inference"
    assert len(result["rendered_prompt_sha256"]) == 64
    assert len(result["rubric_artifact_sha256"]) == 64
    assert len(result["scores"]) == 3
    assert result["scores"][0]["score"] == 2.3
    assert result["scores"][1]["status"] == "not_inferable"
    assert model.runnable.prompt is not None
    assert "private-document-id" not in model.runnable.prompt


def test_inference_preserves_score_when_evidence_offsets_are_wrong() -> None:
    feedback = "Feedback 2.3"
    response = _valid_response(feedback)
    response["scores"][0]["feedback_evidence"][0].update(
        {"start_character": 0, "end_character": 3}
    )
    model = FakeChatModel(response)
    document = ReflectionDocument(
        document_id="doc-1",
        segments=[ReflectionSegment(segment_id="s1", text="Reflection passage", order=0)],
    )

    result = infer_feedback_implied_score(
        model=model,
        document=document,
        rubric=EXPERT_RUBRIC,
        human_feedback=feedback,
        human_feedback_id="feedback-1",
        document_id="doc-1",
        enable_langsmith_tracing=False,
    )

    assert result["scores"][0]["score"] == 2.3
    assert result["scores"][0]["status"] == "inferred_from_feedback"
    evidence = result["scores"][0]["feedback_evidence"][0]
    assert evidence["quote"] == "2.3"
    assert evidence["status"] == "offset_mismatch_quote_found_uniquely"
    assert evidence["suggested_start_character"] == feedback.index("2.3")


def test_inference_accepts_missing_and_reversed_offsets_for_manual_review() -> None:
    feedback = "Feedback 2.3 qualitative statement"
    response = _valid_response(feedback)
    response["scores"][0]["feedback_evidence"][0].update(
        {"start_character": 5, "end_character": 2}
    )
    response["scores"][1] = {
        "dimension_id": "UA",
        "status": "inferred_from_feedback",
        "score": 1.7,
        "confidence": "low",
        "rationale": "A qualitative statement supports a tentative score.",
        "feedback_evidence": [
            {"start_character": None, "end_character": None, "quote": "qualitative statement"}
        ],
    }
    model = FakeChatModel(response)
    document = ReflectionDocument(
        document_id="doc-1",
        segments=[ReflectionSegment(segment_id="s1", text="Reflection passage", order=0)],
    )

    result = infer_feedback_implied_score(
        model=model,
        document=document,
        rubric=EXPERT_RUBRIC,
        human_feedback=feedback,
        human_feedback_id="feedback-1",
        document_id="doc-1",
        enable_langsmith_tracing=False,
    )

    assert result["scores"][0]["score"] == 2.3
    assert result["scores"][0]["feedback_evidence"][0]["status"] == "invalid_offsets"
    assert result["scores"][1]["score"] == 1.7
    assert result["scores"][1]["feedback_evidence"][0]["status"] == "offsets_not_provided"
