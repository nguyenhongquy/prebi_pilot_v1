from __future__ import annotations

from reflection_assessment_feedback.experiment_config import load_experiment_config
from reflection_assessment_feedback.rubric_registry import resolve_rubric

_GENERATION_CONFIG = load_experiment_config()["generation"]
RUBRIC_ARTIFACT = resolve_rubric(
    _GENERATION_CONFIG["rubric_id"],
    _GENERATION_CONFIG["expected_rubric_version"],
)
EXPERT_RUBRIC = RUBRIC_ARTIFACT.rubric
