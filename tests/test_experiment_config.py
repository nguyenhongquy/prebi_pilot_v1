from reflection_assessment_feedback import EXPERT_RUBRIC
from reflection_assessment_feedback.experiment_config import (
    experiment_config_sha256,
    load_experiment_config,
)
from reflection_assessment_feedback.prompt import PROMPT_VERSION
from reflection_assessment_feedback.prompt_registry import resolve_prompt


def test_shared_experiment_config_matches_runtime_versions():
    config = load_experiment_config()

    assert config["config_version"] == 1
    assert config["dataset"]["exploratory_document_id"] == "188"
    generation_prompt = resolve_prompt(
        config["generation"]["prompt_id"],
        config["generation"]["expected_prompt_version"],
    )
    assert generation_prompt.version == PROMPT_VERSION
    assert config["generation"]["expected_rubric_version"] == EXPERT_RUBRIC.version
    assert config["generation"]["enable_langsmith_tracing"] is True
    assert len(experiment_config_sha256()) == 64
