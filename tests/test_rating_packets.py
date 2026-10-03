from __future__ import annotations

import pytest

from reflection_assessment_feedback.experiment_config import load_experiment_config
from reflection_assessment_feedback.rating_packets import build_rating_packet
from reflection_assessment_feedback.prompt_registry import resolve_prompt, render_prompt
from reflection_assessment_feedback.rubric import RUBRIC_ARTIFACT


def _segments() -> list[dict[str, object]]:
    return [{"segment_id": "segment-a", "order": 0, "text": "Synthetic reflection passage."}]


def _run() -> dict[str, object]:
    return {
        "run_id": "opaque-run-id",
        "condition": "G2",
        "repetition": 1,
        "model_name": "private-model-name",
        "output": {
            "assessment": {
                "component_assessments": [
                    {
                        "dimension_id": "SW",
                        "band": 2.1,
                        "justification": "Synthetic rationale.",
                        "evidence_segment_ids": ["segment-a"],
                    }
                ]
            },
            "feedback": {
                "strengths": [{"message": "Synthetic strength.", "evidence_segment_ids": ["segment-a"]}],
                "weaknesses": [],
                "suggestions": [],
            },
        },
    }


def test_assessment_and_feedback_packets_are_blinded_and_task_specific() -> None:
    config = load_experiment_config()["prompts"]
    rubric = RUBRIC_ARTIFACT.rubric.model_dump(mode="json")
    assessment_packet = build_rating_packet(
        task="assessment_quality",
        run=_run(),
        reflection_segments=_segments(),
        rubric=rubric,
    )
    feedback_packet = build_rating_packet(
        task="feedback_quality",
        run=_run(),
        reflection_segments=_segments(),
        rubric=rubric,
    )

    for packet in (assessment_packet, feedback_packet):
        assert packet["condition_blinded"] is True
        assert "condition" not in packet
        assert "model_name" not in packet
        assert "document_id" not in packet
        assert packet["human_protocol"]
        assert packet["displayed_reflection_sha256"]
        assert packet["display_output_sha256"]

    assert assessment_packet["task"] == "assessment_quality"
    assert assessment_packet["protocol_id"] == config["assessment_quality_human_id"]
    assert "dimensions" in assessment_packet["output"]
    assert assessment_packet["displayed_context"] == [
        "segmented_reflection", "rubric", "assessment_output"
    ]
    assert feedback_packet["task"] == "feedback_quality"
    assert feedback_packet["protocol_id"] == config["feedback_quality_human_id"]
    assert feedback_packet["output"]["strengths"][0]["text"] == "Synthetic strength."
    for packet in (assessment_packet, feedback_packet):
        assert packet["protocol_version"] == config[f"{packet['task']}_human_version"]
        assert packet["scale"]["min"] == 1
        assert packet["scale"]["max"] == 3
        for criterion in packet["criteria"]:
            assert set(criterion["anchors"]) == {"1", "2", "3"}
    assert [criterion["criterion_id"] for criterion in feedback_packet["criteria"]] == [
        "correctness", "developmental_usefulness"
    ]
    assert assessment_packet["protocol_version"] == "0.3.0"
    assert [criterion["criterion_id"] for criterion in assessment_packet["criteria"]] == [
        "score_rubric_fit", "evidence_support", "justification_quality"
    ]


def test_legacy_quality_protocol_keeps_original_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_experiment_config()
    config["prompts"]["feedback_quality_human_version"] = "0.1.0"
    monkeypatch.setattr(
        "reflection_assessment_feedback.rating_packets.load_experiment_config", lambda: config
    )
    packet = build_rating_packet(
        task="feedback_quality", run=_run(), reflection_segments=_segments(),
        rubric=RUBRIC_ARTIFACT.rubric.model_dump(mode="json"),
    )
    assert packet["protocol_version"] == "0.1.0"
    assert packet["scale"]["max"] == 5
    assert len(packet["criteria"]) == 5


@pytest.mark.parametrize("task", ["assessment_quality", "feedback_quality"])
def test_revised_quality_llm_prompts_are_registered_and_renderable(task: str) -> None:
    config = load_experiment_config()["prompts"]
    artifact = resolve_prompt(config[f"{task}_llm_id"], config[f"{task}_llm_version"])
    rendered, digest = render_prompt(artifact, packet_json="{}")
    assert artifact.version == ("0.3.0" if task == "assessment_quality" else "0.2.0")
    assert "$packet_json" not in rendered
    assert len(digest) == 64


def test_assessment_v02_preserves_four_criteria(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_experiment_config()
    config["prompts"]["assessment_quality_human_version"] = "0.2.0"
    monkeypatch.setattr(
        "reflection_assessment_feedback.rating_packets.load_experiment_config", lambda: config
    )
    packet = build_rating_packet(
        task="assessment_quality", run=_run(), reflection_segments=_segments(),
        rubric=RUBRIC_ARTIFACT.rubric.model_dump(mode="json"),
    )
    assert packet["protocol_version"] == "0.2.0"
    assert [criterion["criterion_id"] for criterion in packet["criteria"]] == [
        "rubric_alignment", "score_defensibility", "evidence_support", "justification_quality"
    ]
    assert packet["scale"]["max"] == 3
