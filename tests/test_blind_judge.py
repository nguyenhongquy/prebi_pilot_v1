import copy
import json
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

import pytest

import reflection_assessment_feedback.judge_study as study
from reflection_assessment_feedback.judge_stability import analyze_judge_study
from reflection_assessment_feedback.llm_judge import (
    AssessmentAlignmentJudgeOutput, build_judge_input, judge_input_sha256, run_quality_judge,
)
from reflection_assessment_feedback.models import DocumentGenerationOutput
from reflection_assessment_feedback.prompt_registry import resolve_prompt
from scripts import run_blind_g3_playground as blind
from scripts.regenerate_g3_playground import assessment_packet, prepare as prepare_generation
from test_alignment_judge import output as alignment_output
from test_alignment_judge import packet as alignment_packet
from test_document_pipeline import output_record
from test_g3_generation_diagnostic import source_study
from test_llm_judge import FakeChatModel, _packet


def test_blind_payload_and_provenance_exclude_references_not_just_prompt_guidance():
    packet = alignment_packet()
    packet["reference_teacher_feedback"] = "SECRET_TEACHER_REFERENCE"
    packet["gold_analysis_summary"]["private_marker"] = "SECRET_GOLD_ANNOTATION"
    packet["evidence_synthesis"] = "SECRET_GENERATOR_SYNTHESIS"
    model = FakeChatModel(alignment_output())
    judged = run_quality_judge(
        model=model, packet=packet, provider="test", model_name="fake",
        assessment_prompt_version="0.8.0",
    )
    assert model.schema is AssessmentAlignmentJudgeOutput
    assert model.runnable.prompt is not None
    assert "SECRET_" not in model.runnable.prompt
    payload = json.loads(model.runnable.prompt.split("PAKET:\n")[1])
    assert set(payload) == {"reflection", "rubric", "output"}
    assert judged["teacher_reference_sha256"] is None
    assert judged["gold_analysis_summary_sha256"] is None
    before = judge_input_sha256(packet, assessment_prompt_version="0.8.0")
    del packet["reference_teacher_feedback"], packet["gold_analysis_summary"]
    assert judge_input_sha256(packet, assessment_prompt_version="0.8.0") == before
    with pytest.raises(ValueError, match="missing"):
        build_judge_input(packet)
    with pytest.raises(ValueError, match="assessment-quality"):
        build_judge_input(_packet("feedback_quality"), assessment_prompt_version="0.8.0")
    del packet["reflection"]
    with pytest.raises(ValueError, match="requires"):
        build_judge_input(packet, assessment_prompt_version="0.8.0")


def test_blind_prompt_preserves_scoring_rules():
    informed = resolve_prompt("assessment_quality_llm_judge", "0.7.0").text
    blinded = resolve_prompt("assessment_quality_llm_judge", "0.8.0").text
    assert informed.split("\n\n", 1)[1] == blinded.split("\n\n", 1)[1]


def test_seven_output_study_freezes_compares_and_resumes(source_study, monkeypatch):
    root, source = source_study
    original = study.load_judge_study(source)
    original["protocol"]["assessment_prompt_version"] = "0.7.0"
    entry = original["protocol"]["entries"][0]
    entry["identity"].update(cohort_id="synthetic", repetition=1)
    packet = entry["packet"]
    packet["output"] = assessment_packet(
        packet, DocumentGenerationOutput.model_validate(output_record()), packet["packet_id"],
    )["output"]
    baselines = {}

    def informed_result(p):
        return run_quality_judge(
            model=FakeChatModel(alignment_output(1.5)), packet=p, provider="test",
            model_name="fake", assessment_prompt_version="0.7.0",
        )

    original_judgment = informed_result(packet)
    for rep in range(1, 4):
        baselines[(packet["packet_id"], rep)] = {"judgment": copy.deepcopy(original_judgment)}
    monkeypatch.setattr(blind, "study_results", lambda _: (original, baselines))
    for version in ("2.1.0", "2.2.0"):
        directory, protocol = prepare_generation(root, source, g3_supplemental_version=version)
        for rep in range(1, 4):
            record = {
                "protocol_sha256": directory.name, "repetition": rep, "condition": "G3",
                "model_name": protocol["original_model"], "output": output_record(),
                "rendered_prompt_sha256": protocol["rendered_prompt_sha256"],
                "prompt_components": protocol["prompt_components"],
            }
            study._write_once(directory / f"generation-{rep}.json", record)
            p = assessment_packet(
                protocol["source_entry"]["packet"],
                DocumentGenerationOutput.model_validate(record["output"]),
                str(uuid5(NAMESPACE_URL, f"{directory.name}:G3:{rep}")),
            )
            judged = informed_result(p)
            judged.update(
                protocol_sha256=directory.name, generation_repetition=rep,
                source_generation_sha256=study.content_hash(record),
                judge_model=protocol["judge_config"]["model_name"],
            )
            study._write_once(directory / f"alignment-{rep}.json", judged)
    before = {p: p.read_bytes() for p in root.rglob("*.json")}
    directory = blind.prepare(root, source)
    assert blind.prepare(root, source) == directory
    assert study.study_progress(directory)["planned_judgments"] == 21
    manifest = study.load_judge_study(directory)
    assert len(manifest["protocol"]["entries"]) == 7
    for item in manifest["protocol"]["entries"]:
        assert set(item["packet"]) == {
            "task", "packet_id", "rubric_version", "reflection", "rubric", "output",
        }
        assert "reference_teacher_feedback" not in item["rendered_prompt"]
    calls = []

    def judge(**kwargs):
        calls.append(kwargs["packet"]["packet_id"])
        return run_quality_judge(**{
            **kwargs, "model": FakeChatModel(alignment_output(1.5)),
            "enable_langsmith_tracing": False,
        })

    monkeypatch.setattr(study, "run_quality_judge", judge)
    monkeypatch.setattr(study, "reserve_request_slot", lambda *args: None)
    monkeypatch.setattr(blind, "study_results", study.study_results)
    for _ in range(2):
        study.run_judge_study(
            directory, approve_external_processing=True, approve_full_batch=True,
            model=SimpleNamespace(temperature=None),
        )
    assert len(calls) == 21
    assert {p: p.read_bytes() for p in before} == before
    rows = blind.comparison(directory)
    assert len(rows) == 21
    assert all(len(row["reference_blind"]) == 3 for row in rows)
    assert sorted({len(row["reference_informed"]) for row in rows}) == [1, 3]
    assert len(analyze_judge_study(directory)["dimension_decisions"]) == 63
