from __future__ import annotations

import csv
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from reflection_assessment_feedback import (
    EXPERT_RUBRIC,
    GenerationOutput,
    GenerationRequest,
    IntermediateAnalysis,
    ReflectionDocument,
    ReflectionSegment,
    RubricGenerationRunner,
    SegmentAnalysis,
    eligible_document_ids,
    load_human_annotated_document,
    make_generation_request,
)
from reflection_assessment_feedback.generator import output_for_review, validate_output
from reflection_assessment_feedback import generator as generator_module
from reflection_assessment_feedback.models import AnalysisSource, Condition
from reflection_assessment_feedback.prompt import build_prompt


def _document() -> ReflectionDocument:
    return ReflectionDocument(
        document_id="synthetic-doc",
        segments=[ReflectionSegment(segment_id="s1", text="Synthetic text.", order=0)],
    )


def _analysis(source: AnalysisSource = "human") -> IntermediateAnalysis:
    return IntermediateAnalysis(
        source=source,
        segments=[
            SegmentAnalysis(
                segment_id="s1",
                scope="in_scope",
                component_bands={"SW": "2", "UA": "1", "HA": "0"},
            )
        ],
    )


def _output() -> GenerationOutput:
    return GenerationOutput.model_validate(
        {
            "assessment": {
                "component_assessments": [
                    {
                        "dimension_id": dimension_id,
                        "band": band,
                        "justification": "Synthetic justification.",
                        "evidence_segment_ids": ["s1"],
                    }
                    for dimension_id, band in (("SW", 2), ("UA", 1), ("HA", 0))
                ]
            },
            "feedback": {
                "strengths": [{"message": "Synthetic strength.", "evidence_segment_ids": ["s1"]}],
                "weaknesses": [],
                "suggestions": [{"message": "Synthetic suggestion.", "evidence_segment_ids": ["s1"]}],
            },
        }
    )


def test_request_contract_accepts_g1_g2_and_g3() -> None:
    document = _document()
    g1 = make_generation_request(
        condition="G1", document=document, rubric=EXPERT_RUBRIC
    )
    g2 = make_generation_request(
        condition="G2",
        document=document,
        rubric=EXPERT_RUBRIC,
        analysis=_analysis("predicted"),
    )
    g3 = make_generation_request(
        condition="G3",
        document=document,
        rubric=EXPERT_RUBRIC,
        analysis=_analysis("human"),
    )
    assert (g1.condition, g2.condition, g3.condition) == ("G1", "G2", "G3")


@pytest.mark.parametrize(
    ("condition", "analysis"),
    [
        ("G1", _analysis("human")),
        ("G2", _analysis("human")),
        ("G3", None),
    ],
)
def test_request_contract_rejects_wrong_analysis(
    condition: Condition,
    analysis: IntermediateAnalysis | None,
) -> None:
    with pytest.raises(ValidationError):
        GenerationRequest(
            condition=condition,
            document=_document(),
            rubric=EXPERT_RUBRIC,
            analysis=analysis,
        )


def test_out_of_scope_is_not_band_zero() -> None:
    with pytest.raises(ValidationError):
        SegmentAnalysis(
            segment_id="s1",
            scope="out_of_scope",
            component_bands={"SW": "0", "UA": "N", "HA": "N"},
        )


def test_analysis_summary_uses_only_in_scope_segments() -> None:
    analysis = IntermediateAnalysis(
        source="human",
        segments=[
            SegmentAnalysis(
                segment_id="s1",
                scope="in_scope",
                component_bands={"SW": "2", "UA": "1", "HA": "0"},
            ),
            SegmentAnalysis(
                segment_id="s2",
                scope="in_scope",
                component_bands={"SW": "3", "UA": "2", "HA": "1"},
            ),
            SegmentAnalysis(
                segment_id="s3",
                scope="out_of_scope",
                component_bands={"SW": "N", "UA": "N", "HA": "N"},
            ),
        ],
    )

    summary = {item.dimension_id: item for item in analysis.summary or []}
    assert summary["SW"].positive_band_mean == 2.5
    assert summary["SW"].positive_segment_count == 2
    assert summary["SW"].positive_segment_percentage == 100.0
    assert summary["UA"].positive_band_mean == 1.5
    assert summary["SW"].positive_band_percentages == {
        "1": 0.0,
        "2": 50.0,
        "3": 50.0,
    }
    assert summary["SW"].in_scope_segment_count == 2


def test_analysis_summary_excludes_zero_without_penalizing_other_dimensions() -> None:
    analysis = IntermediateAnalysis(
        source="human",
        segments=[
            SegmentAnalysis(
                segment_id="s1", scope="in_scope",
                component_bands={"SW": "0", "UA": "2", "HA": "0"},
            ),
            SegmentAnalysis(
                segment_id="s2", scope="in_scope",
                component_bands={"SW": "3", "UA": "0", "HA": "0"},
            ),
        ],
    )
    summary = {item.dimension_id: item for item in analysis.summary or []}
    assert summary["SW"].positive_band_mean == 3.0
    assert summary["SW"].positive_segment_count == 1
    assert summary["SW"].positive_segment_percentage == 50.0
    assert summary["UA"].positive_band_mean == 2.0
    assert summary["HA"].positive_band_mean is None
    assert summary["HA"].positive_segment_percentage == 0.0
    assert summary["HA"].positive_band_percentages is None
    assert summary["SW"].positive_band_percentages == {
        "1": 0.0, "2": 0.0, "3": 100.0
    }


def test_positive_band_mean_for_sparse_ua_example() -> None:
    bands = ["0"] * 11 + ["1"] * 11 + ["2"] * 12 + ["3"] * 3
    analysis = IntermediateAnalysis(
        source="human",
        segments=[
            SegmentAnalysis(
                segment_id=f"s{index}",
                scope="in_scope",
                component_bands={"SW": "0", "UA": band, "HA": "0"},
            )
            for index, band in enumerate(bands)
        ],
    )
    summary = {item.dimension_id: item for item in analysis.summary or []}
    assert summary["UA"].in_scope_segment_count == 37
    assert summary["UA"].positive_segment_count == 26
    assert summary["UA"].positive_segment_percentage == 70.27
    assert summary["UA"].positive_band_mean == 1.69
    assert summary["UA"].positive_band_percentages == {
        "1": 42.31, "2": 46.15, "3": 11.54
    }
    assert summary["SW"].positive_band_mean is None


def test_prompt_includes_supplemental_analysis_only_when_available() -> None:
    document = _document()
    g1_prompt = build_prompt(
        make_generation_request(condition="G1", document=document, rubric=EXPERT_RUBRIC)
    )
    g2_prompt = build_prompt(
        make_generation_request(
            condition="G2",
            document=document,
            rubric=EXPERT_RUBRIC,
            analysis=_analysis("predicted"),
        )
    )

    assert "Ergänzende Analyse berücksichtigen" not in g1_prompt
    assert "Ergänzende Analyse berücksichtigen" in g2_prompt
    g1_payload = json.loads(g1_prompt.split("Arbeitsgrundlage (JSON):\n", maxsplit=1)[1])
    g2_payload = json.loads(g2_prompt.split("Arbeitsgrundlage (JSON):\n", maxsplit=1)[1])
    assert "segment_analysis" not in g1_payload
    assert "analysis" not in g1_payload["reflection"]["segments"][0]
    assert g2_payload["reflection"]["segments"][0]["analysis"] == {
        "scope": "in_scope",
        "component_bands": {"SW": "2", "UA": "1", "HA": "not_present"},
    }
    assert "0" in g2_payload["rubric"]["dimensions"][0]["bands"]
    assert _analysis("predicted").segments[0].component_bands["HA"] == "0"
    assert g2_payload["segment_analysis"]["source"] == "predicted"
    assert "segments" not in g2_payload["segment_analysis"]
    assert "high_band_segments" not in g2_prompt
    assert g2_payload["segment_analysis"]["summary"][0]["positive_band_mean"] == 2.0
    summary = {item["dimension_id"]: item for item in g2_payload["segment_analysis"]["summary"]}
    assert "band_percentages" not in summary["SW"]
    assert summary["SW"]["positive_band_percentages"] == {
        "1": 0.0, "2": 100.0, "3": 0.0
    }
    assert summary["HA"]["positive_band_percentages"] is None


def test_prompt_includes_in_scope_band_three_reflection_segments() -> None:
    document = ReflectionDocument(
        document_id="synthetic-doc",
        segments=[
            ReflectionSegment(
                segment_id="s1",
                text="I reconsidered my approach and changed my next lesson plan.",
                order=0,
            )
        ],
    )
    analysis = IntermediateAnalysis(
        source="human",
        segments=[
            SegmentAnalysis(
                segment_id="s1",
                scope="in_scope",
                component_bands={"SW": "3", "UA": "2", "HA": "1"},
            )
        ],
    )
    request = make_generation_request(
        condition="G3",
        document=document,
        rubric=EXPERT_RUBRIC,
        analysis=analysis,
    )

    prompt = build_prompt(request)

    payload = json.loads(prompt.split("Arbeitsgrundlage (JSON):\n", maxsplit=1)[1])
    segments = payload["reflection"]["segments"]
    assert segments == [
        {
            "segment_id": "s1",
            "order": 0,
            "text": "I reconsidered my approach and changed my next lesson plan.",
            "analysis": {
                "scope": "in_scope",
                "component_bands": {"SW": "3", "UA": "2", "HA": "1"},
            },
        }
    ]
    assert payload["segment_analysis"]["summary"][0]["positive_band_mean"] == 3.0


def test_prompt_joins_annotations_in_document_order() -> None:
    document = ReflectionDocument(
        document_id="synthetic-doc",
        segments=[
            ReflectionSegment(segment_id="s1", text="First passage", order=0),
            ReflectionSegment(segment_id="s2", text="Second passage", order=1),
        ],
    )
    analysis = IntermediateAnalysis(
        source="human",
        segments=[
            SegmentAnalysis(
                segment_id="s2", scope="out_of_scope",
                component_bands={"SW": "N", "UA": "N", "HA": "N"},
            ),
            SegmentAnalysis(
                segment_id="s1", scope="in_scope",
                component_bands={"SW": "0", "UA": "2", "HA": "0"},
            ),
        ],
    )
    request = make_generation_request(
        condition="G3", document=document, rubric=EXPERT_RUBRIC, analysis=analysis
    )
    prompt = build_prompt(request)
    payload = json.loads(prompt.split("Arbeitsgrundlage (JSON):\n", maxsplit=1)[1])
    segments = payload["reflection"]["segments"]
    assert [segment["segment_id"] for segment in segments] == ["s1", "s2"]
    assert segments[0]["analysis"]["component_bands"] == {
        "SW": "not_present", "UA": "2", "HA": "not_present"
    }
    assert segments[1]["analysis"]["scope"] == "out_of_scope"


def test_dimension_assessment_accepts_decimal_scores() -> None:
    result = _output().model_dump(mode="python")
    result["assessment"]["component_assessments"][0]["band"] = 2.1

    parsed = GenerationOutput.model_validate(result)

    assert parsed.assessment.component_assessments[0].band == 2.1


def test_output_validation_rejects_unknown_evidence() -> None:
    result = _output()
    result.assessment.component_assessments[0].evidence_segment_ids = ["not-in-document"]
    request = make_generation_request(
        condition="G1", document=_document(), rubric=EXPERT_RUBRIC
    )
    with pytest.raises(ValueError, match="segment ID"):
        validate_output(request, result)


def test_review_output_derives_evidence_orders_from_document() -> None:
    document = ReflectionDocument(
        document_id="synthetic-doc",
        segments=[
            ReflectionSegment(segment_id="s1", text="First", order=0),
            ReflectionSegment(segment_id="s2", text="Second", order=7),
        ],
    )
    output = _output()
    output.assessment.component_assessments[0].evidence_segment_ids = ["s2", "s1"]

    review = output_for_review(output, document)

    assessment = review["assessment"]["component_assessments"][0]
    assert assessment["evidence_segment_ids"] == ["s2", "s1"]
    assert assessment["evidence_segment_orders"] == [7, 0]
    assert "evidence_segment_orders" not in output.model_dump(mode="json")["assessment"]["component_assessments"][0]


def test_review_output_rejects_unknown_evidence_id() -> None:
    output = _output()
    output.assessment.component_assessments[0].evidence_segment_ids = ["unknown"]
    with pytest.raises(ValueError, match="segment ID"):
        output_for_review(output, _document())


class FakeRunnable:
    def __init__(self) -> None:
        self.config: dict[str, Any] | None = None

    def invoke(self, input: str, config: dict[str, Any] | None = None) -> GenerationOutput:
        assert "Synthetic text." in input
        self.config = config
        return _output()


class FakeChatModel:
    model = "fake-model"
    temperature = 0.0

    def __init__(self) -> None:
        self.runnable = FakeRunnable()

    def with_structured_output(self, schema: type) -> FakeRunnable:
        assert schema is GenerationOutput
        return self.runnable


def test_shared_runner_adds_condition_trace_metadata() -> None:
    model = FakeChatModel()
    runner = RubricGenerationRunner(model)
    request = make_generation_request(
        condition="G3",
        document=_document(),
        rubric=EXPERT_RUBRIC,
        analysis=_analysis("human"),
    )
    run = runner.generate(request, repetition=2)
    assert run.condition == "G3"
    assert run.repetition == 2
    assert run.model_name == "fake-model"
    assert run.generation_parameters == {"temperature": 0.0}
    assert run.feedback_letter.startswith("## Stärken")
    assert model.runnable.config is not None
    assert model.runnable.config["metadata"]["condition"] == "G3"


def test_runner_fails_closed_when_ambient_tracing_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    model = FakeChatModel()
    request = make_generation_request(
        condition="G1", document=_document(), rubric=EXPERT_RUBRIC
    )
    with pytest.raises(RuntimeError, match="tracing is active"):
        RubricGenerationRunner(model).generate(request)
    assert model.runnable.config is None


@pytest.mark.parametrize(
    ("api_key", "project", "missing_setting"),
    [
        (None, "test-project", "LANGSMITH_API_KEY"),
        ("test-key", None, "LANGSMITH_PROJECT"),
    ],
)
def test_enabled_tracing_requires_api_key_and_project(
    monkeypatch: pytest.MonkeyPatch,
    api_key: str | None,
    project: str | None,
    missing_setting: str,
) -> None:
    for name, value in (
        ("LANGSMITH_API_KEY", api_key),
        ("LANGSMITH_PROJECT", project),
        ("LANGSMITH_ENDPOINT", "https://eu.api.smith.langchain.com"),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    model = FakeChatModel()
    request = make_generation_request(
        condition="G1", document=_document(), rubric=EXPERT_RUBRIC
    )

    with pytest.raises(RuntimeError, match=missing_setting):
        RubricGenerationRunner(model, enable_langsmith_tracing=True).generate(request)
    assert model.runnable.config is None


def test_enabled_tracing_uses_configured_langsmith_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://eu.api.smith.langchain.com")
    model = FakeChatModel()
    request = make_generation_request(
        condition="G1", document=_document(), rubric=EXPERT_RUBRIC
    )
    context_options: dict[str, Any] = {}
    client_options: dict[str, Any] = {}
    accessed_projects: list[str | None] = []
    flushed_clients: list[object] = []

    class FakeTracingClient:
        def read_project(self, *, project_name: str) -> None:
            accessed_projects.append(project_name)

        def flush(self) -> None:
            flushed_clients.append(self)

    def fake_client(**kwargs: Any) -> object:
        client_options.update(kwargs)
        return FakeTracingClient()

    monkeypatch.setattr(generator_module, "Client", fake_client)

    def fake_tracing_context(**kwargs: Any) -> Any:
        context_options.update(kwargs)
        return nullcontext()

    monkeypatch.setattr(generator_module, "tracing_context", fake_tracing_context)

    run = RubricGenerationRunner(model, enable_langsmith_tracing=True).generate(request)

    assert run.condition == "G1"
    assert context_options["enabled"] is True
    assert context_options["project_name"] == "test-project"
    assert client_options == {
        "api_url": "https://eu.api.smith.langchain.com",
        "api_key": "test-key",
    }
    assert accessed_projects == ["test-project"]
    assert len(flushed_clients) == 1
    assert model.runnable.config is not None
    assert "callbacks" not in model.runnable.config


def test_tracing_rejects_non_eu_endpoint_before_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    model = FakeChatModel()
    request = make_generation_request(
        condition="G1", document=_document(), rubric=EXPERT_RUBRIC
    )

    with pytest.raises(RuntimeError, match="requires LANGSMITH_ENDPOINT"):
        RubricGenerationRunner(model, enable_langsmith_tracing=True).generate(request)
    assert model.runnable.config is None


def test_g3_loader_uses_synthetic_annotated_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "synthetic.csv"
    columns = [
        "segment_id",
        "candidate_id",
        "text",
        "document_id",
        "segment_order",
        "scope_target",
        "scope_target_known",
        "requires_review",
        "target_situationserfassung",
        "target_analyse",
        "target_konsequenzen",
        "known_situationserfassung",
        "known_analyse",
        "known_konsequenzen",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerow(
            {
                "segment_id": "s1",
                "candidate_id": "c1",
                "text": "Synthetic reflection segment.",
                "document_id": "d1",
                "segment_order": "0",
                "scope_target": "0",
                "scope_target_known": "true",
                "requires_review": "false",
                "target_situationserfassung": "",
                "target_analyse": "",
                "target_konsequenzen": "",
                "known_situationserfassung": "false",
                "known_analyse": "false",
                "known_konsequenzen": "false",
            }
        )

    document, analysis = load_human_annotated_document(csv_path, "d1")
    assert analysis.segments[0].component_bands == {
        "SW": "N",
        "UA": "N",
        "HA": "N",
    }
    assert eligible_document_ids(csv_path) == ["d1"]


def test_g3_loader_rejects_alternative_candidates(tmp_path: Path) -> None:
    csv_path = tmp_path / "ambiguous.csv"
    base = {
        "segment_id": "s1",
        "text": "Synthetic reflection segment.",
        "document_id": "d1",
        "segment_order": "0",
        "scope_target": "1",
        "scope_target_known": "true",
        "requires_review": "false",
        "target_situationserfassung": "2",
        "target_analyse": "1",
        "target_konsequenzen": "0",
        "known_situationserfassung": "true",
        "known_analyse": "true",
        "known_konsequenzen": "true",
    }
    columns = ["candidate_id", *base]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerow({"candidate_id": "c1", **base})
        writer.writerow({"candidate_id": "c2", **base})

    with pytest.raises(ValueError, match="alternative candidate"):
        load_human_annotated_document(csv_path, "d1")