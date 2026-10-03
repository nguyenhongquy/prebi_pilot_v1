from __future__ import annotations

import json
from pathlib import Path

import pytest

from reflection_assessment_feedback.experiment_config import load_experiment_config
from reflection_assessment_feedback import (
    EXPERT_RUBRIC,
    HumanAnalysisCandidate,
    HumanAnalysisContext,
    HumanSegmentAnalysis,
    IntermediateAnalysis,
    ReflectionDocument,
    ReflectionSegment,
    SegmentAnalysis,
    make_generation_request,
)
from reflection_assessment_feedback.prompt import prompt_components_for_request
from reflection_assessment_feedback.prompt_registry import (
    MANIFEST_PATH,
    PromptArtifact,
    render_prompt,
    resolve_prompt,
)
from reflection_assessment_feedback.rubric import RUBRIC_ARTIFACT
from reflection_assessment_feedback.rubric_registry import resolve_rubric


def test_all_registered_prompt_hashes_match_library_files() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    for item in manifest["artifacts"]:
        artifact = resolve_prompt(item["id"], item["version"])
        assert artifact.sha256 == item["sha256"]
        assert artifact.status == item["status"]


def test_experiment_config_pins_both_modalities_for_each_rating_task() -> None:
    config = load_experiment_config()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    registered = {(item["id"], item["version"]) for item in manifest["artifacts"]}

    for task in ("feedback_quality", "feedback_implied_score"):
        human_id = config["prompts"][f"{task}_human_id"]
        human_version = config["prompts"][f"{task}_human_version"]
        llm_id = config["prompts"][f"{task}_llm_id"]
        llm_version = config["prompts"][f"{task}_llm_version"]
        assert (human_id, human_version) in registered
        assert (llm_id, llm_version) in registered


def test_template_render_requires_exact_variables() -> None:
    config = load_experiment_config()
    artifact = resolve_prompt(
        config["prompts"]["feedback_quality_llm_id"],
        config["prompts"]["feedback_quality_llm_version"],
    )

    with pytest.raises(ValueError, match="variables mismatch"):
        render_prompt(artifact)

    rendered, digest = render_prompt(artifact, packet_json='{"packet_id":"opaque"}')
    assert '"packet_id":"opaque"' in rendered
    assert len(digest) == 64


def test_registered_generation_prompt_matches_runtime_prompt() -> None:
    config = load_experiment_config()["generation"]
    artifact = resolve_prompt(config["prompt_id"], config["expected_prompt_version"])
    assert artifact.task == "generation"
    assert artifact.modality == "llm_prompt"
    assert artifact.status == "active"


def test_generation_prompt_components_are_condition_specific() -> None:
    document = ReflectionDocument(
        document_id="registry-test",
        segments=[ReflectionSegment(segment_id="s1", text="Example reflection.", order=0)],
    )
    predicted = IntermediateAnalysis(
        source="predicted",
        segments=[SegmentAnalysis(segment_id="s1", scope="in_scope", component_bands={"SW": "1", "UA": "2", "HA": "0"})],
    )
    observed_human = HumanAnalysisContext(
        segments=[
            HumanSegmentAnalysis(
                segment_id="s1",
                candidates=[
                    HumanAnalysisCandidate(
                        candidate_id="c1",
                        scope="in_scope",
                        scope_known=True,
                        raw_scope_value="1",
                        component_bands={"SW": "1", "UA": None, "HA": "0"},
                        raw_component_values={"SW": "1", "UA": "", "HA": "0"},
                        known_component_bands={"SW": True, "UA": False, "HA": True},
                        requires_review=True,
                    )
                ],
            )
        ]
    )
    g1 = make_generation_request(condition="G1", document=document, rubric=EXPERT_RUBRIC)
    g2 = make_generation_request(
        condition="G2", document=document, rubric=EXPERT_RUBRIC, analysis=predicted
    )
    g3 = make_generation_request(
        condition="G3", document=document, rubric=EXPERT_RUBRIC, analysis=observed_human
    )

    assert [artifact.artifact_id for artifact in prompt_components_for_request(g1)] == ["generation"]
    assert [artifact.artifact_id for artifact in prompt_components_for_request(g2)] == [
        "generation", "generation_g2_predicted_analysis"
    ]
    assert [artifact.artifact_id for artifact in prompt_components_for_request(g3)] == [
        "generation", "generation_g3_observed_human_analysis"
    ]


def test_configured_rubric_matches_immutable_rubric_artifact() -> None:
    config = load_experiment_config()["generation"]
    artifact = resolve_rubric(config["rubric_id"], config["expected_rubric_version"])

    assert artifact.sha256 == RUBRIC_ARTIFACT.sha256
    assert artifact.rubric.model_dump(mode="json") == EXPERT_RUBRIC.model_dump(mode="json")
