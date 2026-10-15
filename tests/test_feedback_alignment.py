import copy
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import reflection_assessment_feedback.judge_study as study
from reflection_assessment_feedback.judge_stability import analyze_judge_study
from reflection_assessment_feedback.llm_judge import (
    FeedbackQualityJudgeOutput, FeedbackReferenceAlignmentOutput, judge_output_model,
    run_quality_judge, validate_feedback_alignment,
)
from test_llm_judge import FakeChatModel, _packet
from test_judge_study import frozen
from test_judge_batch import setup_batch


def packet():
    result = _packet("feedback_quality")
    result["reference_teacher_feedback"] = "You describe the situation clearly. Explain the reasons."
    result["output"] = {
        "strengths": [{"text": "Your description is clear.", "evidence_segment_ids": ["s1"]}],
        "weaknesses": [],
        "suggestions": [{"text": "Explain why this happened.", "evidence_segment_ids": ["s1"]}],
    }
    return result


def comparison(relationship="agreement", central=True, grounding="supported"):
    return {
        "relationship": relationship, "central": central, "grounding": grounding,
        "human_excerpt": "You describe the situation clearly.",
        "generated_references": [{"component": "strengths", "item_index": 0, "excerpt": "Your description is clear."}],
        "rationale": "Same substantive observation.", "evidence_segment_ids": ["s1"],
    }


def output(verdict="aligned", comparisons=None):
    return {
        "verdict": verdict, "rationale": "Synthetic reference alignment.",
        "comparisons": [comparison()] if comparisons is None else comparisons,
        "overall_comment": "Synthetic.",
    }


def test_versioned_runner_payload_and_exact_quote_validation():
    assert judge_output_model("feedback_quality", "0.3.0") is FeedbackQualityJudgeOutput
    model = FakeChatModel(output())
    result = run_quality_judge(
        model=model, packet=packet(), provider="test", model_name="fake",
        feedback_prompt_version="0.4.0",
    )
    assert model.schema is FeedbackReferenceAlignmentOutput
    assert result["prompt_version"] == "0.4.0"
    assert model.runnable.prompt is not None
    payload = json.loads(model.runnable.prompt.split("PAKET:\n")[1])
    assert set(payload) == {"reflection", "rubric", "output", "reference_teacher_feedback"}
    assert "gold_analysis_summary" not in model.runnable.prompt
    assert "ratings" not in result["result"]
    for change in ("human", "generated", "index", "segment"):
        data = output()
        point = data["comparisons"][0]
        if change == "human":
            point["human_excerpt"] = "Invented reference"
        elif change == "generated":
            point["generated_references"][0]["excerpt"] = "Invented generated quote"
        elif change == "index":
            point["generated_references"][0]["item_index"] = 99
        else:
            point["evidence_segment_ids"] = ["unknown"]
        with pytest.raises(ValueError):
            validate_feedback_alignment(FeedbackReferenceAlignmentOutput.model_validate(data), packet())


def test_alignment_distinguishes_omissions_additions_and_reference_fallibility():
    omission = comparison("central_omission", grounding="not_applicable")
    omission["generated_references"] = []
    partial = FeedbackReferenceAlignmentOutput.model_validate(output("partially_aligned", [omission]))
    validate_feedback_alignment(partial, packet())
    with pytest.raises(ValidationError):
        FeedbackReferenceAlignmentOutput.model_validate(output("misaligned", [omission]))
    with pytest.raises(ValidationError):
        FeedbackReferenceAlignmentOutput.model_validate(output("aligned", [omission]))
    addition = comparison("compatible_addition")
    addition["human_excerpt"] = None
    FeedbackReferenceAlignmentOutput.model_validate(output(comparisons=[comparison(), addition]))
    reference_uncertainty = comparison("reference_uncertainty", grounding="not_applicable")
    reference_uncertainty["generated_references"] = []
    FeedbackReferenceAlignmentOutput.model_validate(output("partially_aligned", [reference_uncertainty]))
    conflict = comparison("contradiction", grounding="unsupported")
    FeedbackReferenceAlignmentOutput.model_validate(output("misaligned", [conflict]))
    FeedbackReferenceAlignmentOutput.model_validate(output("unable_to_judge", []))
    with pytest.raises(ValidationError):
        FeedbackReferenceAlignmentOutput.model_validate(output(comparisons=[]))


def test_quote_validation_allows_only_whitespace_equivalence():
    p = packet()
    p["reference_teacher_feedback"] = p["reference_teacher_feedback"].replace(" ", "\u00a0")
    p["output"]["strengths"][0]["text"] = "Your\ndescription is  clear."
    validate_feedback_alignment(FeedbackReferenceAlignmentOutput.model_validate(output()), p)
    for quote in ("Your description is unclear.", "your description is clear.", "Your description ... clear."):
        data = output()
        data["comparisons"][0]["generated_references"][0]["excerpt"] = quote
        with pytest.raises(ValueError, match="Generated excerpt"):
            validate_feedback_alignment(FeedbackReferenceAlignmentOutput.model_validate(data), p)


def test_feedback_study_freezes_fresh_schema_runs_and_resumes(frozen):
    original, root, _, calls, config = frozen
    document = study.load_judge_study(original)["protocol"]["entries"][0]["identity"]["document_id"]
    before = copy.deepcopy(config)
    directory = study.freeze_judge_study(
        root, cohort="playground", tasks=("feedback_quality",), import_existing=False,
        feedback_prompt_version="0.4.0", source_study=original, document_ids=(document,),
    )
    assert config == before
    assert study.study_progress(directory)["planned_judgments"] == 9
    assert set(analyze_judge_study(directory)["decisions"].verdict) == {"missing"}

    def judge(**kwargs):
        p = kwargs["packet"]
        point = comparison()
        point["human_excerpt"] = p["reference_teacher_feedback"]
        point["generated_references"] = [{
            "component": "strengths", "item_index": 0,
            "excerpt": p["output"]["strengths"][0]["text"],
        }]
        point["evidence_segment_ids"] = [p["reflection"]["segments"][0]["segment_id"]]
        return run_quality_judge(**{
            **kwargs, "model": FakeChatModel(output(comparisons=[point])),
            "enable_langsmith_tracing": False,
        })

    calls.side_effect = judge
    for _ in range(2):
        study.run_judge_study(
            directory, approve_external_processing=True, approve_full_batch=True,
            model=SimpleNamespace(temperature=None),
        )
    assert calls.call_count == 9
    tables = analyze_judge_study(directory)
    assert len(tables["decisions"]) == 9
    assert tables["stability"].unanimous.all()
    assert tables["stability"].pairwise_agreements.sum() == 9
    manifest = study.load_judge_study(directory)
    e = manifest["protocol"]["entries"][0]
    path = study._result_path(directory, e["packet"]["packet_id"], 2)
    record = json.loads(path.read_text())
    record["judgment"]["result"]["verdict"] = "partially_aligned"
    path.write_text(json.dumps(record))
    assert not analyze_judge_study(directory)["stability"].unanimous.all()
    study._result_path(directory, e["packet"]["packet_id"], 3).unlink()
    assert analyze_judge_study(directory)["stability"].missing.sum() == 1
    with pytest.raises(ValueError, match="fresh judgments"):
        study.freeze_judge_study(root, tasks=("feedback_quality",), feedback_prompt_version="0.4.0")


def test_single_feedback_call_per_output_is_not_a_stability_estimate(frozen):
    original, root, _, _, _ = frozen
    document = study.load_judge_study(original)["protocol"]["entries"][0]["identity"]["document_id"]
    directory = study.freeze_judge_study(
        root, cohort="playground", tasks=("feedback_quality",), import_existing=False,
        feedback_prompt_version="0.4.0", judge_repetitions=1,
        source_study=original, document_ids=(document,),
    )
    assert study.study_progress(directory)["planned_judgments"] == 3
    assert analyze_judge_study(directory)["stability"].unanimous.isna().all()
    with pytest.raises(ValueError, match="at least two"):
        study.freeze_judge_study(root, judge_repetitions=1)
