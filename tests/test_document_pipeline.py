from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from reflection_assessment_feedback import EXPERT_RUBRIC, make_generation_request
from reflection_assessment_feedback.evidence_profiles import build_evidence_profiles
from reflection_assessment_feedback.generator import validate_output
from reflection_assessment_feedback.generation_consistency import document_score_agreement
from reflection_assessment_feedback.models import (
    DOCUMENT_SCORES, DocumentGenerationOutput, GenerationOutput,
    HumanAnalysisCandidate, HumanAnalysisContext, HumanSegmentAnalysis,
    IntermediateAnalysis, ReflectionDocument, ReflectionSegment, SegmentAnalysis,
)
from reflection_assessment_feedback.prompt import build_prompt
from reflection_assessment_feedback.rating_packets import build_rating_packet, build_reference_rating_packet
from reflection_assessment_feedback.rubric_registry import resolve_rubric


def output_record() -> dict:
    claim = {"interpretation": "A specific observation is connected to an explanation.", "evidence_refs": ["s1"]}
    return {
        "evidence_synthesis": [
            {
                "dimension_id": dimension,
                "demonstrated_depth": [claim],
                "development": [claim],
                "integrative_coherence": [claim],
                "limitations": [],
                "uncertainty": [],
            } for dimension in ("SW", "UA", "HA")
        ],
        "assessment": {"component_assessments": [
            {
                "dimension_id": dimension, "band": 1.5,
                "justification": "Basic performance is demonstrated, with some supported elaboration.",
                "evidence_segment_ids": ["s1"],
                "intermediate_score_explanation": "The basic anchor is met; contextual support is present but insufficiently developed for the next anchor.",
            } for dimension in ("SW", "UA", "HA")
        ]},
        "feedback": {
            "strengths": [{"message": "Die Beobachtung ist konkret.", "evidence_segment_ids": ["s1"]}],
            "weaknesses": [],
            "suggestions": [{"message": "Begründen Sie die Verbindung genauer.", "evidence_segment_ids": ["s1"]}],
        },
    }


def document() -> ReflectionDocument:
    return ReflectionDocument(document_id="doc", segments=[
        ReflectionSegment(segment_id="s1", text="An observation and an explanation.", order=0),
        ReflectionSegment(segment_id="s2", text="Another perspective.", order=1),
    ])


def human_context() -> HumanAnalysisContext:
    def candidate(identifier: str, band: str) -> HumanAnalysisCandidate:
        return HumanAnalysisCandidate(
            candidate_id=identifier, scope="in_scope", scope_known=True, raw_scope_value="1",
            component_bands={"SW": band, "UA": None, "HA": "0"},
            raw_component_values={"SW": band, "UA": "", "HA": "0"},
            known_component_bands={"SW": True, "UA": False, "HA": True},
            requires_review=True,
        )
    return HumanAnalysisContext(segments=[
        HumanSegmentAnalysis(segment_id="s1", candidates=[candidate("c1", "1"), candidate("c2", "3"), candidate("c3", "1")]),
        HumanSegmentAnalysis(segment_id="s2", candidates=[candidate("c4", "2")]),
    ])


@pytest.mark.parametrize("score", DOCUMENT_SCORES)
def test_document_score_enumeration(score: float) -> None:
    record = output_record()
    record["assessment"]["component_assessments"][0]["band"] = score
    assert DocumentGenerationOutput.model_validate(record).assessment.component_assessments[0].band == score


@pytest.mark.parametrize("score", [-1, 0.1, 1.1, 2.49, 2.9, 3.1, float("nan"), True, "1.5"])
def test_invalid_document_scores_are_rejected_not_rounded(score: object) -> None:
    record = output_record()
    record["assessment"]["component_assessments"][0]["band"] = score
    with pytest.raises(ValidationError):
        DocumentGenerationOutput.model_validate(record)


def test_half_step_requires_anchor_gap_explanation() -> None:
    record = output_record()
    record["assessment"]["component_assessments"][0]["intermediate_score_explanation"] = None
    with pytest.raises(ValidationError, match="Intermediate"):
        DocumentGenerationOutput.model_validate(record)


def test_provider_schema_enumerates_numeric_scores_explicitly() -> None:
    schema = DocumentGenerationOutput.model_json_schema()
    assert schema["$defs"]["DocumentScore"]["enum"] == list(DOCUMENT_SCORES)
    assert schema["$defs"]["DocumentScore"]["type"] == "number"


def test_profiles_preserve_conflicts_and_count_segments_separately() -> None:
    analysis = human_context()
    before = analysis.model_dump(mode="json")
    profile = build_evidence_profiles(make_generation_request(
        condition="G3", document=document(), rubric=EXPERT_RUBRIC, analysis=analysis,
    ))
    sw, ua, ha = profile["dimensions"]
    assert sw["distinct_assigned_segment_count"] == 2
    assert sw["assigned_candidate_annotation_count"] == 4
    assert sw["candidate_band_counts"] == {"A": 2, "B": 1, "C": 1}
    assert sw["distinct_segment_band_counts"] == {"A": 1, "B": 1, "C": 1}
    assert sw["passages"][0]["conflicting_annotations"]
    assert [c["candidate_id"] for c in sw["passages"][0]["candidates"]] == ["c1", "c2", "c3"]
    assert ua["passages"][0]["candidates"][0]["status"] == "unknown"
    assert ha["passages"][0]["candidates"][0]["status"] == "not_assigned"
    assert analysis.model_dump(mode="json") == before


def test_all_conditions_share_rubric_and_no_numerical_summary_inputs() -> None:
    predicted = IntermediateAnalysis(source="predicted", segments=[
        SegmentAnalysis(segment_id="s1", scope="in_scope", component_bands={"SW": "1", "UA": "2", "HA": "3"}),
        SegmentAnalysis(segment_id="s2", scope="out_of_scope", component_bands={"SW": "N", "UA": "N", "HA": "N"}),
    ])
    before = predicted.model_dump(mode="json")
    for condition, analysis in (("G1", None), ("G2", predicted), ("G3", human_context())):
        request = make_generation_request(condition=condition, document=document(), rubric=EXPERT_RUBRIC, analysis=analysis)
        prompt = build_prompt(request)
        payload = json.loads(prompt.split("Arbeitsgrundlage (JSON):\n")[1])
        assert payload["rubric"]["version"] == "1.2.0"
        assert payload["rubric"]["assessment_principles"] == EXPERT_RUBRIC.assessment_principles
        for forbidden in ("positive_band_mean", "mean_positive_band", "positive_band_percentages"):
            assert forbidden not in prompt
        assert ("evidence_profile" in payload) == (condition != "G1")
        if condition == "G2":
            dimensions = payload["evidence_profile"]["dimensions"]
            assert [d["passages"][0]["candidates"][0]["label"] for d in dimensions] == ["A", "B", "C"]
            assert all(d["passages"][1]["candidates"][0]["label"] == "NA" for d in dimensions)
    assert predicted.model_dump(mode="json") == before


def test_synthesis_contract_has_no_score_fields_and_checks_evidence() -> None:
    record = output_record()
    record["evidence_synthesis"][0]["score"] = 2
    with pytest.raises(ValidationError, match="extra_forbidden"):
        DocumentGenerationOutput.model_validate(record)
    record = output_record()
    record["evidence_synthesis"][0]["development"][0] = {
        "interpretation": "Unsupported", "evidence_refs": ["unknown"],
    }
    output = DocumentGenerationOutput.model_validate(record)
    request = make_generation_request(condition="G1", document=document(), rubric=EXPERT_RUBRIC)
    with pytest.raises(ValueError, match="segment ID"):
        validate_output(request, output)


def test_historical_tenth_scores_still_readable_without_migration() -> None:
    record = output_record()
    record.pop("evidence_synthesis")
    for assessment in record["assessment"]["component_assessments"]:
        assessment["band"] = 2.1
        assessment.pop("intermediate_score_explanation")
    original = copy.deepcopy(record)
    output = GenerationOutput.model_validate(record)
    old_rubric = resolve_rubric("unterrichtsreflexion-expert", "1.0.0").rubric
    validate_output(make_generation_request(condition="G1", document=document(), rubric=old_rubric), output)
    assert output.model_dump(mode="json") == original


def test_packet_synthesis_is_separate_and_hidden_without_protocol_approval() -> None:
    output = output_record()
    run = {"output": output, "pipeline_version": "2.0.0", "rubric_version": "1.2.0"}
    segments = [s.model_dump() for s in document().segments]
    for task in ("assessment_quality", "feedback_quality"):
        packet = build_rating_packet(task=task, run=run, reflection_segments=segments, rubric=EXPERT_RUBRIC.model_dump(mode="json"))
        assert packet["evidence_synthesis"] == output["evidence_synthesis"]
        assert packet["synthesis_display_allowed"] is False
        assert "evidence_synthesis" not in packet["output"]
        assert len(packet["display_synthesis_sha256"]) == 64
    reference = build_reference_rating_packet(reflection_segments=segments, rubric=EXPERT_RUBRIC.model_dump(mode="json"), reflection_id="reflection")
    assert "evidence_synthesis" not in reference
    assert "output" not in reference
    assert reference["rubric"]["assessment_principles"] == EXPERT_RUBRIC.assessment_principles
    with pytest.raises(ValueError, match="historical"):
        build_rating_packet(task="assessment_quality", run={"output": output}, reflection_segments=segments, rubric=EXPERT_RUBRIC.model_dump(mode="json"))


def test_document_agreement_uses_half_step_categories() -> None:
    result = document_score_agreement([0, 1, 2, 3], [0, 1.5, 2, 2])
    assert result["mae"] == 0.375
    assert result["exact_agreement"] == 0.5
    assert result["within_0_5_agreement"] == 0.75
    assert result["weighted_cohen_kappa"] is not None
    with pytest.raises(ValueError):
        document_score_agreement([1], [1.1])
    assert document_score_agreement([1], [1])["weighted_cohen_kappa"] is None


def test_dataset_schema_preserves_historical_scores_but_restricts_new_instrument() -> None:
    from jsonschema import Draft202012Validator

    schema = json.loads((Path(__file__).resolve().parents[1] / "data/assessment_feedback_dataset.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    definition = {"$ref": "#/$defs/generationRun", "$defs": schema["$defs"]}
    validator = Draft202012Validator(definition)
    record = output_record()
    run = {
        "run_id": "r", "document_id": "doc", "condition": "G1", "repetition": 1,
        "model_provider": "test", "model_name": "fake", "prompt_version": "2.0.0",
        "rubric_version": "1.2.0", "pipeline_version": "2.0.0", "created_at": "2026-01-01T00:00:00Z",
        "analysis_source": "none", "analysis_reference_id": None, "validation_status": "valid",
        "evidence_synthesis": record["evidence_synthesis"],
        "assessment": {"dimensions": {
            item["dimension_id"]: {
                "score": item["band"], "justification": item["justification"],
                "evidence_segment_ids": item["evidence_segment_ids"],
                "intermediate_score_explanation": item["intermediate_score_explanation"],
            } for item in record["assessment"]["component_assessments"]
        }},
        "feedback": {key: [{"text": p["message"], "evidence_segment_ids": p["evidence_segment_ids"]} for p in points] for key, points in record["feedback"].items()},
    }
    validator.validate(run)
    run["assessment"]["dimensions"]["SW"]["score"] = 1.1
    assert list(validator.iter_errors(run))
    historical = copy.deepcopy(run)
    historical["rubric_version"] = "1.0.0"
    historical.pop("pipeline_version")
    historical.pop("evidence_synthesis")
    validator.validate(historical)
    run["assessment"]["dimensions"]["SW"]["score"] = 1.5
    run["assessment"]["dimensions"]["SW"]["intermediate_score_explanation"] = "  "
    assert list(validator.iter_errors(run))
    run["assessment"]["dimensions"]["SW"]["intermediate_score_explanation"] = "Anchor gap"
    run["evidence_synthesis"][0]["dimension_id"] = "UA"
    assert list(validator.iter_errors(run))


def test_nested_and_normalized_generation_forms_export_equivalent_packets() -> None:
    from jsonschema import Draft202012Validator

    output = output_record()
    versions = {"pipeline_version": "2.0.0", "rubric_version": "1.2.0"}
    nested = {**versions, "output": output}
    normalized = {
        **versions,
        "evidence_synthesis": output["evidence_synthesis"],
        "assessment": {"dimensions": {
            item["dimension_id"]: {key: value for key, value in item.items() if key not in ("band", "dimension_id")}
            | {"score": item["band"]} for item in output["assessment"]["component_assessments"]
        }},
        "feedback": {group: [{"text": p["message"], "evidence_segment_ids": p["evidence_segment_ids"]} for p in points] for group, points in output["feedback"].items()},
    }
    schema = json.loads((Path(__file__).resolve().parents[1] / "data/assessment_feedback_dataset.schema.json").read_text())
    validator = Draft202012Validator({"$ref": "#/$defs/ratingPacket", "$defs": schema["$defs"]})
    packets = [
        build_rating_packet(task="assessment_quality", run=run, reflection_segments=[s.model_dump() for s in document().segments], rubric=EXPERT_RUBRIC.model_dump(mode="json"))
        for run in (nested, normalized)
    ]
    for packet in packets:
        validator.validate(packet)
    assert packets[0]["output"] == packets[1]["output"]
    assert packets[0]["display_synthesis_sha256"] == packets[1]["display_synthesis_sha256"]
    packets[0]["output"]["dimensions"][0]["score"] = 1.1
    assert list(validator.iter_errors(packets[0]))
