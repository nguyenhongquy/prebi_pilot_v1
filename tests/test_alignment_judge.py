import copy
import json
import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from reflection_assessment_feedback.llm_judge import (
    AssessmentAlignmentJudgeOutput, AssessmentQualityJudgeOutput, judge_output_model,
    run_quality_judge, validate_alignment_evidence,
)
from reflection_assessment_feedback.judge_stability import analyze_judge_study
import reflection_assessment_feedback.judge_study as study
from test_llm_judge import FakeChatModel, _packet
from test_judge_study import frozen
from test_judge_batch import setup_batch
from scripts import run_alignment_playground as alignment_cli


def decision(dimension="SW", score: float = 2):
    return {
        "dimension_id": dimension, "generated_score": score, "verdict": "defensible",
        "rationale": "Supported by the reflection and anchor.",
        "rubric_requirement": "Concrete situation-related evidence.",
        "evidence_segment_ids": ["s1"],
        "half_step_check": ({
            "lower_anchor_met": True, "upper_anchor_features_present": True,
            "remaining_gap_supported": True, "rationale": "Specific remaining gap.",
        } if score in (0.5, 1.5, 2.5) else None),
        "explanation_diagnostic": "Explanation is brief; this does not invalidate the score.",
    }


def output(score: float = 2):
    return {"dimensions": [decision(d, score) for d in ("SW", "UA", "HA")],
            "overall_comment": "Synthetic alignment result."}


def packet(score=2):
    result = _packet("assessment_quality")
    result["output"]["dimensions"] = [
        {"dimension_id": d, "score": score, "justification": "Brief.",
         "evidence_segment_ids": ["s1"], "intermediate_score_explanation": None}
        for d in ("SW", "UA", "HA")
    ]
    return result


@pytest.mark.parametrize(
    ("arguments", "cohort", "documents", "source"),
    [
        (["--cohort", "remaining-test"], "remaining-test", None, None),
        (["--source-study", "/protected/source"], "playground", ("3",), "/protected/source"),
    ],
)
def test_alignment_cli_preserves_pilot_and_selects_full_remaining_cohort(
    monkeypatch, tmp_path, arguments, cohort, documents, source,
):
    monkeypatch.setattr(sys, "argv", ["alignment", *arguments])
    monkeypatch.setenv("PREBI_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(alignment_cli, "load_dotenv", lambda *args, **kwargs: None)
    seen = []

    def freeze(root, **kwargs):
        seen.append(kwargs)
        return tmp_path

    class Table:
        def to_string(self, **kwargs):
            return "Synthetic table."

    monkeypatch.setattr(alignment_cli, "freeze_judge_study", freeze)
    monkeypatch.setattr(alignment_cli, "study_progress", lambda _: {"saved_judgments": 0})
    monkeypatch.setattr(alignment_cli, "analyze_alignment_study", lambda _: {
        "dimension_stability": Table(), "verdict_distribution": Table(),
    })
    alignment_cli.main()
    assert len(seen) == 1
    assert seen[0]["cohort"] == cohort
    assert seen[0]["document_ids"] == documents
    assert (str(seen[0]["source_study"]) if source else seen[0]["source_study"]) == source
    assert seen[0]["conditions"] == ("G1", "G2", "G3")
    assert seen[0]["tasks"] == ("assessment_quality",)
    assert seen[0]["assessment_prompt_version"] == "0.7.0"
    assert seen[0]["judge_repetitions"] == 3
    assert seen[0]["import_existing"] is False


@pytest.mark.parametrize("score", [0, 0.5, 1, 1.5, 2, 2.5, 3])
def test_alignment_prompt_schema_and_all_score_boundaries(score):
    model = FakeChatModel(output(score))
    result = run_quality_judge(
        model=model, packet=packet(score), provider="test", model_name="fake",
        assessment_prompt_version="0.7.0",
    )
    assert model.schema is AssessmentAlignmentJudgeOutput
    assert result["prompt_version"] == "0.7.0"
    assert "ratings" not in result["result"]
    assert len(result["result"]["dimensions"]) == 3
    assert "0.5, 1.5 oder 2.5" in model.runnable.prompt
    assert judge_output_model("assessment_quality", "0.6.0") is AssessmentQualityJudgeOutput


@pytest.mark.parametrize("field", ["lower_anchor_met", "upper_anchor_features_present", "remaining_gap_supported"])
def test_half_step_violation_cannot_be_marked_defensible(field):
    data = output(1.5)
    data["dimensions"][0]["half_step_check"][field] = False
    with pytest.raises(ValidationError, match="every rubric check"):
        AssessmentAlignmentJudgeOutput.model_validate(data)
    data["dimensions"][0]["verdict"] = "not_defensible"
    assert AssessmentAlignmentJudgeOutput.model_validate(data)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "unknown_dimension", "bad_score", "missing_check", "extra_check"])
def test_invalid_decisions_fail(mutation):
    data = output(1.5 if mutation == "missing_check" else 2)
    if mutation == "duplicate":
        data["dimensions"][1]["dimension_id"] = "SW"
    elif mutation == "missing":
        data["dimensions"].pop()
    elif mutation == "unknown_dimension":
        data["dimensions"][0]["dimension_id"] = "OTHER"
    elif mutation == "bad_score":
        data["dimensions"][0]["generated_score"] = 1.25
    elif mutation == "missing_check":
        data["dimensions"][0]["half_step_check"] = None
    else:
        data["dimensions"][0]["half_step_check"] = decision(score=1.5)["half_step_check"]
    with pytest.raises(ValidationError):
        AssessmentAlignmentJudgeOutput.model_validate(data)


def test_evidence_and_generated_score_are_validated():
    data = output()
    data["dimensions"][0]["evidence_segment_ids"] = ["unknown"]
    with pytest.raises(ValueError, match="unknown reflection segment"):
        validate_alignment_evidence(AssessmentAlignmentJudgeOutput.model_validate(data), packet())
    data = output()
    data["dimensions"][0]["generated_score"] = 3
    with pytest.raises(ValueError, match="changed the generated score"):
        validate_alignment_evidence(AssessmentAlignmentJudgeOutput.model_validate(data), packet())


def test_unable_is_distinct_from_missing_and_explanation_quality():
    data = output(1.5)
    data["dimensions"][0].update(
        verdict="unable_to_judge", half_step_check=None, evidence_segment_ids=[],
        rationale="Required reflection content missing.",
    )
    parsed = AssessmentAlignmentJudgeOutput.model_validate(data)
    assert parsed.dimensions[1].verdict == "defensible"


@pytest.mark.parametrize("version", ["0.7.0", "0.8.0"])
def test_frozen_alignment_reuses_exact_inputs_and_resumes(frozen, monkeypatch, version):
    original_directory, root, _, calls, config = frozen
    original = study.load_judge_study(original_directory)
    document_id = original["protocol"]["entries"][0]["identity"]["document_id"]
    before = copy.deepcopy(config)
    directory = study.freeze_judge_study(
        root, cohort="playground", tasks=("assessment_quality",),
        assessment_prompt_version=version, import_existing=False,
        source_study=original_directory, document_ids=(document_id,),
    )
    current = study.load_judge_study(directory)
    assert len(current["protocol"]["entries"]) == 3
    assert config == before
    assert study.study_progress(directory)["planned_judgments"] == 9
    originals = {e["packet"]["packet_id"]: e for e in original["protocol"]["entries"]}
    for e in current["protocol"]["entries"]:
        if version == "0.7.0":
            assert e["input_sha256"] == originals[e["packet"]["packet_id"]]["input_sha256"]
        else:
            assert e["input_sha256"] != originals[e["packet"]["packet_id"]]["input_sha256"]
            assert e["expected_metadata"]["teacher_reference_sha256"] is None
            assert e["expected_metadata"]["gold_analysis_summary_sha256"] is None
    assert set(analyze_judge_study(directory)["dimension_decisions"].verdict) == {"missing"}
    with pytest.raises(RuntimeError, match="approval"):
        study.run_judge_study(directory, approve_external_processing=True)

    def judge(**kwargs):
        p = kwargs["packet"]
        data = output()
        for dimension in data["dimensions"]:
            source = next(d for d in p["output"]["dimensions"] if d["dimension_id"] == dimension["dimension_id"])
            dimension.update(decision(dimension["dimension_id"], source["score"]))
            dimension["evidence_segment_ids"] = [p["reflection"]["segments"][0]["segment_id"]]
        return run_quality_judge(
            **{**kwargs, "model": FakeChatModel(data), "enable_langsmith_tracing": False},
        )

    calls.side_effect = judge
    for _ in range(2):
        study.run_judge_study(
            directory, approve_external_processing=True, approve_full_batch=True,
            model=SimpleNamespace(temperature=None),
        )
    assert calls.call_count == 9
    tables = analyze_judge_study(directory)
    assert len(tables["dimension_decisions"]) == 27
    assert tables["dimension_stability"].unanimous.all()
    assert tables["pairwise_agreement"].exact_agreement.all()

    entry = current["protocol"]["entries"][0]
    identifier = entry["packet"]["packet_id"]
    second = study._result_path(directory, identifier, 2)
    record = json.loads(second.read_text())
    record["judgment"]["result"]["dimensions"][0]["verdict"] = "not_defensible"
    record["judgment"]["result"]["dimensions"][1].update(
        verdict="unable_to_judge", half_step_check=None, evidence_segment_ids=[],
    )
    second.write_text(json.dumps(record))
    changed = analyze_judge_study(directory)
    condition = entry["identity"]["condition"]
    stable = changed["dimension_stability"].set_index(["condition", "dimension_id"])
    assert stable.loc[(condition, "SW"), "defensible_to_not_defensible"]
    assert stable.loc[(condition, "UA"), "unable_to_judge"] == 1
    assert stable.loc[(condition, "UA"), "unanimous"] is None
    assert changed["pairwise_agreement"].comparable.sum() == 25
    study._result_path(directory, identifier, 3).unlink()
    missing = analyze_judge_study(directory)["dimension_stability"]
    assert missing.missing_judgments.sum() == 3
    with pytest.raises(ValueError, match="missing"):
        study.freeze_judge_study(
            root, cohort="playground", tasks=("assessment_quality",),
            assessment_prompt_version="0.7.0", import_existing=False,
            source_study=original_directory, document_ids=("missing-document",),
        )
