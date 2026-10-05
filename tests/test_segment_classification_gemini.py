from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from langsmith.utils import tracing_is_enabled

from reflection_assessment_feedback.models import ReflectionDocument, ReflectionSegment
from reflection_assessment_feedback.segment_classification_gemini import (
    GeminiReflectionPrediction,
    SegmentClassifierGemini,
)


class FakeRunnable:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = iter(responses)
        self.prompts: list[str] = []
        self.tracing_states: list[bool] = []

    def invoke(self, prompt: str) -> dict[str, Any]:
        self.tracing_states.append(bool(tracing_is_enabled()))
        self.prompts.append(prompt)
        return next(self.responses)


class FakeChatModel:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.runnable = FakeRunnable(responses)

    def with_structured_output(self, schema: type) -> FakeRunnable:
        assert schema is GeminiReflectionPrediction
        return self.runnable


def _prediction(scope: str, band: str) -> dict[str, Any]:
    value = band if scope == "in_scope" else "N"
    return {
        "scope": scope,
        "component_bands": {"SW": value, "UA": value, "HA": value},
    }


def _reflection_output(*predictions: dict[str, Any]) -> dict[str, Any]:
    return {
        "predictions": [
            {"index": index, **prediction}
            for index, prediction in enumerate(predictions)
        ]
    }


def test_predict_document_sends_whole_reflection_and_maps_indices() -> None:
    model = FakeChatModel([{
        "predictions": [
            {"index": 1, "scope": "out_of_scope", "component_bands": {"SW": "0", "UA": "0", "HA": "0"}},
            {"index": 0, **_prediction("in_scope", "2")},
        ],
    }])
    classifier = SegmentClassifierGemini(model)
    document = ReflectionDocument(document_id="doc", segments=[
        ReflectionSegment(segment_id="s1", text="Opening reflection", order=0),
        ReflectionSegment(segment_id="s2", text="Later reflection", order=1),
    ])

    analysis = classifier.predict_document(document)

    assert analysis.source == "predicted"
    assert [segment.segment_id for segment in analysis.segments] == ["s1", "s2"]
    assert analysis.segments[0].component_bands == {"SW": "2", "UA": "2", "HA": "2"}
    assert analysis.segments[1].scope == "out_of_scope"
    assert analysis.segments[1].component_bands == {"SW": "N", "UA": "N", "HA": "N"}
    assert len(model.runnable.prompts) == 1
    assert "vollstaendige deutsche Reflexion als ein Dokument" in model.runnable.prompts[0]
    assert "Opening reflection" in model.runnable.prompts[0]
    assert "Later reflection" in model.runnable.prompts[0]


def test_predict_frame_assembles_full_documents_and_expands_candidate_rows() -> None:
    model = FakeChatModel([
        _reflection_output(
            _prediction("out_of_scope", "0"),
            _prediction("in_scope", "1"),
        ),
        _reflection_output(_prediction("in_scope", "3")),
    ])
    classifier = SegmentClassifierGemini(model)
    rows = pd.DataFrame([
        {"candidate_id": "a1", "document_id": "d1", "segment_id": "s1", "segment_order": 1,
         "text": "Second passage", "scope_target": 1},
        {"candidate_id": "a2", "document_id": "d1", "segment_id": "s1", "segment_order": 1,
         "text": "Second passage", "scope_target": 1},
        {"candidate_id": "b1", "document_id": "d1", "segment_id": "s0", "segment_order": 0,
         "text": "First passage", "scope_target": 0},
        {"candidate_id": "c1", "document_id": "d2", "segment_id": "s2", "segment_order": 0,
         "text": "Another reflection", "scope_target": 1},
    ])
    rows.index = [0, 0, 1, 0]

    predictions = classifier.predict_frame(rows)

    assert predictions.scope.tolist() == [1, 1, 0, 1]
    assert predictions.SW.tolist() == ["1", "1", "N", "3"]
    assert len(model.runnable.prompts) == 2
    assert model.runnable.prompts[0].index("First passage") < model.runnable.prompts[0].index("Second passage")


def test_gold_scope_prediction_receives_whole_reflection_and_forces_only_gold_segments() -> None:
    model = FakeChatModel([_reflection_output(
        _prediction("in_scope", "2"),
        _prediction("out_of_scope", "0"),
    )])
    classifier = SegmentClassifierGemini(model)
    rows = pd.DataFrame([
        {"candidate_id": "a", "document_id": "d", "segment_id": "s1", "segment_order": 0,
         "text": "Gold in-scope passage", "scope_target": 1},
        {"candidate_id": "b", "document_id": "d", "segment_id": "s2", "segment_order": 1,
         "text": "Other passage", "scope_target": 0},
    ])

    predictions = classifier.predict_frame(rows, gold_in_scope=True)

    assert predictions.scope.tolist() == [1, 0]
    assert "Gold in-scope passage" in model.runnable.prompts[0]
    assert "Other passage" in model.runnable.prompts[0]
    assert '"scope_constraint": "in_scope"' in model.runnable.prompts[0]


def test_evaluate_uses_whole_reflections_for_end_to_end_and_gold_scope_reports() -> None:
    model = FakeChatModel([
        _reflection_output(
            _prediction("in_scope", "1"),
            _prediction("out_of_scope", "0"),
        ),
        _reflection_output(
            _prediction("in_scope", "1"),
            _prediction("out_of_scope", "0"),
        ),
    ])
    rows = pd.DataFrame([
        {"candidate_id": "a", "document_id": "d", "segment_id": "s1", "segment_order": 0,
         "text": "In-scope passage", "scope_target": 1,
         "target_situationserfassung": 1, "target_analyse": 1, "target_konsequenzen": 1},
        {"candidate_id": "b", "document_id": "d", "segment_id": "s2", "segment_order": 1,
         "text": "Out-of-scope passage", "scope_target": 0,
         "target_situationserfassung": None, "target_analyse": None, "target_konsequenzen": None},
    ])

    report = SegmentClassifierGemini(model).evaluate(rows)

    assert report["rows"] == 2
    assert 0 <= report["selection_macro_f1"] <= 1
    assert report["reports"]["SW_gold_scope"]["1"]["support"] == 1
    assert report["reports"]["SW_end_to_end"]["N"]["support"] == 1
    assert len(model.runnable.prompts) == 2
    assert all("Out-of-scope passage" in prompt for prompt in model.runnable.prompts)


def test_prediction_rejects_incomplete_or_invalid_document_output() -> None:
    incomplete = FakeChatModel([_reflection_output(_prediction("in_scope", "1"))])
    document = ReflectionDocument(document_id="d", segments=[
        ReflectionSegment(segment_id="s1", text="First", order=0),
        ReflectionSegment(segment_id="s2", text="Second", order=1),
    ])
    with pytest.raises(ValueError, match="each segment index exactly once"):
        SegmentClassifierGemini(incomplete).predict_document(document)

    invalid = FakeChatModel([_reflection_output({
        "scope": "in_scope", "component_bands": {"SW": "1", "UA": "1"},
    })])
    with pytest.raises(ValueError):
        SegmentClassifierGemini(invalid).predict_document(ReflectionDocument(
            document_id="d",
            segments=[ReflectionSegment(segment_id="s1", text="Text", order=0)],
        ))


def test_prediction_rejects_candidate_copies_with_conflicting_segment_content() -> None:
    classifier = SegmentClassifierGemini(FakeChatModel([]))
    rows = pd.DataFrame([
        {"document_id": "d", "segment_id": "s", "segment_order": 0, "text": "First"},
        {"document_id": "d", "segment_id": "s", "segment_order": 0, "text": "Different"},
    ])

    with pytest.raises(ValueError, match="disagree"):
        classifier.predict_frame(rows)


def test_predict_document_disables_inherited_langsmith_tracing(monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    classifier = SegmentClassifierGemini(FakeChatModel([
        _reflection_output(_prediction("in_scope", "1")),
    ]))
    document = ReflectionDocument(document_id="d", segments=[
        ReflectionSegment(segment_id="s", text="Reflection", order=0),
    ])

    classifier.predict_document(document)

    assert classifier._runnable.tracing_states == [False]


def test_predict_document_enables_langsmith_tracing_with_validated_eu_client(monkeypatch) -> None:
    from reflection_assessment_feedback import segment_classification_gemini

    class FakeClient:
        instances = []

        def __init__(self, *, api_url: str, api_key: str) -> None:
            assert api_url == "https://eu.api.smith.langchain.com"
            assert api_key == "test-key"
            self.project: str | None = None
            self.flushed = False
            self.instances.append(self)

        def read_project(self, *, project_name: str) -> None:
            self.project = project_name

        def flush(self) -> None:
            self.flushed = True

    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
    monkeypatch.setattr(segment_classification_gemini, "Client", FakeClient)
    classifier = SegmentClassifierGemini(
        FakeChatModel([_reflection_output(_prediction("in_scope", "1"))]),
        enable_langsmith_tracing=True,
    )
    document = ReflectionDocument(document_id="d", segments=[
        ReflectionSegment(segment_id="s", text="Reflection", order=0),
    ])

    classifier.predict_document(document)

    assert classifier._project_name == "test-project"
    assert classifier._runnable.tracing_states == [True]
    assert FakeClient.instances[0].flushed


def test_enabled_tracing_requires_credentials_and_eu_endpoint(monkeypatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    with pytest.raises(RuntimeError, match="LANGSMITH_API_KEY"):
        SegmentClassifierGemini(FakeChatModel([]), enable_langsmith_tracing=True)

    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    with pytest.raises(RuntimeError, match="eu.api.smith.langchain.com"):
        SegmentClassifierGemini(FakeChatModel([]), enable_langsmith_tracing=True)