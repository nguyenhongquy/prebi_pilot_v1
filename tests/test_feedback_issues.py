import copy
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import reflection_assessment_feedback.judge_study as study
from reflection_assessment_feedback.judge_stability import analyze_judge_study
from reflection_assessment_feedback.llm_judge import (
    FeedbackFidelityJudgeOutput, FeedbackIssueJudgeOutput, FeedbackReferenceFirstJudgeOutput,
    judge_output_model,
    run_quality_judge, validate_feedback_issues,
)
from test_feedback_alignment import packet
from test_llm_judge import FakeChatModel
from test_judge_study import frozen
from test_judge_batch import setup_batch


def output(severity="major"):
    return {
        "reference_concerns": [{
            "concern_id": "c1", "human_excerpt": "Explain the reasons.",
            "centrality_rationale": "Explicit developmental request.",
            "reference_support": "supported", "coverage": "missing",
            "generated_references": [], "evidence_segment_ids": ["s1"],
            "rationale": "Not addressed in the generated feedback.",
        }],
        "issues": [{
            "issue_type": "central_omission", "severity": severity, "concern_ids": ["c1"],
            "human_excerpt": "Explain the reasons.", "generated_references": [],
            "evidence_segment_ids": ["s1"], "rationale": "Central request omitted.",
            "consequence": "The learner misses the requested development priority.",
        }],
        "overall_severity": severity, "rationale": "A central omission is major.",
        "reference_uncertainties": [],
    }


def test_major_rules_aggregation_and_evidence():
    data = output()
    parsed = FeedbackIssueJudgeOutput.model_validate(data)
    validate_feedback_issues(parsed, packet())
    with pytest.raises(ValidationError, match="major"):
        FeedbackIssueJudgeOutput.model_validate(output("minor"))
    for modify in ("overall", "unknown_concern", "unsupported_reference", "missing_issue"):
        changed = copy.deepcopy(data)
        if modify == "overall":
            changed["overall_severity"] = "no_error"
        elif modify == "unknown_concern":
            changed["issues"][0]["concern_ids"] = ["unknown"]
        elif modify == "unsupported_reference":
            changed["reference_concerns"][0]["reference_support"] = "unsupported"
        else:
            changed["issues"] = []
        with pytest.raises(ValidationError):
            FeedbackIssueJudgeOutput.model_validate(changed)
    changed = copy.deepcopy(data)
    changed["issues"][0]["human_excerpt"] = "Invented human quote."
    with pytest.raises(ValueError, match="Human excerpt"):
        validate_feedback_issues(FeedbackIssueJudgeOutput.model_validate(changed), packet())
    changed = copy.deepcopy(data)
    changed["issues"][0]["evidence_segment_ids"] = ["unknown"]
    with pytest.raises(ValueError, match="unknown reflection"):
        validate_feedback_issues(FeedbackIssueJudgeOutput.model_validate(changed), packet())


def test_no_error_minor_and_uncertainty_are_distinct():
    data = output()
    data["reference_concerns"][0]["coverage"] = "addressed"
    data["issues"] = []
    data["overall_severity"] = "no_error"
    FeedbackIssueJudgeOutput.model_validate(data)
    data["reference_concerns"][0]["coverage"] = "unable_to_judge"
    with pytest.raises(ValidationError):
        FeedbackIssueJudgeOutput.model_validate(data)
    data["overall_severity"] = "unable_to_judge"
    FeedbackIssueJudgeOutput.model_validate(data)
    data = output()
    data["reference_concerns"] = []
    data["issues"][0].update(
        issue_type="secondary_omission", severity="minor", concern_ids=[],
    )
    data["overall_severity"] = "minor"
    FeedbackIssueJudgeOutput.model_validate(data)


@pytest.mark.parametrize("version", ["0.5.0", "0.6.0", "0.7.0"])
def test_versioned_runner_and_six_call_resume(frozen, version):
    original, root, _, calls, _ = frozen
    directory = study.freeze_judge_study(
        root, cohort="playground", tasks=("feedback_quality",), import_existing=False,
        feedback_prompt_version=version, judge_repetitions=1, source_study=original,
    )
    assert study.study_progress(directory)["planned_judgments"] == 6
    assert set(analyze_judge_study(directory)["decisions"].overall_severity) == {"missing"}

    def judge(**kwargs):
        p = kwargs["packet"]
        data = output()
        for section in ("reference_concerns", "issues"):
            data[section][0]["human_excerpt"] = p["reference_teacher_feedback"]
            data[section][0]["evidence_segment_ids"] = [p["reflection"]["segments"][0]["segment_id"]]
        return run_quality_judge(**{
            **kwargs, "model": FakeChatModel(data), "enable_langsmith_tracing": False,
        })

    calls.side_effect = judge
    for _ in range(2):
        study.run_judge_study(
            directory, approve_external_processing=True, approve_full_batch=True,
            model=SimpleNamespace(temperature=None),
        )
    assert calls.call_count == 6
    tables = analyze_judge_study(directory)
    assert len(tables["decisions"]) == len(tables["issues"]) == 6
    assert set(tables["decisions"].overall_severity) == {"major"}
    manifest = study.load_judge_study(directory)
    entry = manifest["protocol"]["entries"][0]
    path = study._result_path(directory, entry["packet"]["packet_id"], 1)
    record = json.loads(path.read_text())
    record["judgment"]["result"]["overall_severity"] = "minor"
    path.write_text(json.dumps(record))
    with pytest.raises(ValidationError):
        study.study_results(directory)


def test_fidelity_missing_reference_is_not_automatically_an_error():
    data = output()
    data["issues"] = []
    data["overall_severity"] = "no_error"
    assert judge_output_model("feedback_quality", "0.6.0") is FeedbackFidelityJudgeOutput
    parsed = FeedbackFidelityJudgeOutput.model_validate(data)
    validate_feedback_issues(parsed, packet())
    data["reference_concerns"][0]["coverage"] = "unable_to_judge"
    FeedbackFidelityJudgeOutput.model_validate(data)


@pytest.mark.parametrize("issue_type", ["central_omission", "contradiction"])
def test_fidelity_materiality_not_issue_type_determines_severity(issue_type):
    data = output("minor")
    data["issues"][0]["issue_type"] = issue_type
    if issue_type == "contradiction":
        data["issues"][0]["generated_references"] = [{
            "component": "strengths", "item_index": 0,
            "excerpt": packet()["output"]["strengths"][0]["text"],
        }]
    parsed = FeedbackFidelityJudgeOutput.model_validate(data)
    validate_feedback_issues(parsed, packet())
    data["issues"][0]["human_excerpt"] = None
    data["issues"][0]["concern_ids"] = []
    FeedbackFidelityJudgeOutput.model_validate(data)


@pytest.mark.parametrize("issue_type", ["local_imprecision", "secondary_omission"])
def test_fidelity_local_issues_must_be_minor(issue_type):
    data = output()
    data["issues"][0].update(
        issue_type=issue_type, concern_ids=[], generated_references=[{
            "component": "strengths", "item_index": 0, "excerpt": "Some text",
        }],
    )
    with pytest.raises(ValidationError, match="minor"):
        FeedbackFidelityJudgeOutput.model_validate(data)


def test_fidelity_decisive_uncertainty_and_known_major():
    data = output()
    data["overall_severity"] = "unable_to_judge"
    with pytest.raises(ValidationError, match="documented decisive uncertainty"):
        FeedbackFidelityJudgeOutput.model_validate(data)


    data["reference_uncertainties"] = [{
        "relationship": "reference_uncertainty", "central": False,
        "human_excerpt": None, "generated_references": [], "grounding": "uncertain",
        "evidence_segment_ids": [], "rationale": "A prior version is unavailable.",
    }]
    with pytest.raises(ValidationError, match="major issue"):
        FeedbackFidelityJudgeOutput.model_validate(data)
    data["overall_severity"] = "major"
    FeedbackFidelityJudgeOutput.model_validate(data)
    data["issues"] = []
    data["overall_severity"] = "unable_to_judge"
    FeedbackFidelityJudgeOutput.model_validate(data)


@pytest.mark.parametrize("issue_type", ["central_omission", "secondary_omission", "contradiction"])
def test_reference_first_issues_require_supported_reference(issue_type):
    data = output("minor")
    data["issues"][0]["issue_type"] = issue_type
    if issue_type == "contradiction":
        data["issues"][0]["generated_references"] = [{
            "component": "strengths", "item_index": 0, "excerpt": "Some text",
        }]
    assert judge_output_model("feedback_quality", "0.7.0") is FeedbackReferenceFirstJudgeOutput
    FeedbackReferenceFirstJudgeOutput.model_validate(data)
    data["issues"][0]["concern_ids"] = []
    with pytest.raises(ValidationError, match="reference concerns"):
        FeedbackReferenceFirstJudgeOutput.model_validate(data)
    data["issues"][0]["concern_ids"] = ["c1"]
    data["reference_concerns"][0]["reference_support"] = "uncertain"
    with pytest.raises(ValidationError):
        FeedbackReferenceFirstJudgeOutput.model_validate(data)
    data["issues"] = []
    data["overall_severity"] = "no_error"
    FeedbackReferenceFirstJudgeOutput.model_validate(data)
