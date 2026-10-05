import json
import os
import tempfile
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from reflection_assessment_feedback.experiment_config import load_experiment_config
from reflection_assessment_feedback.prompt_registry import resolve_prompt
from reflection_assessment_feedback.rubric import RUBRIC_ARTIFACT


@pytest.mark.parametrize("reused", [False, True])
def test_saved_g2_runs_reach_provenance_cell(tmp_path, reused):
    notebook_path = Path(__file__).resolve().parents[1] / "04_generate_g2.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    generation = next(source for source in sources if "G2_RUN_RECORDS = []" in source)
    generation = generation[generation.index("G2_RUN_RECORDS = []"):]
    provenance = next(source for source in sources if "for generated_run in g2_runs:" in source)
    config = load_experiment_config()["generation"]
    prompt = resolve_prompt(config["prompt_id"], config["expected_prompt_version"])
    generated = SimpleNamespace(
        run_id="synthetic", repetition=1, model_name="fake", prompt_version=prompt.version,
        generation_parameters={}, prompt_artifact_sha256=prompt.sha256,
        rubric_artifact_sha256=RUBRIC_ARTIFACT.sha256, rendered_prompt_sha256="a" * 64,
        prompt_components=[],
        output=SimpleNamespace(
            assessment=SimpleNamespace(component_assessments=[
                SimpleNamespace(dimension_id=dimension, band=1, justification="synthetic", evidence_segment_ids=[])
                for dimension in ("SW", "UA", "HA")
            ]),
            feedback=SimpleNamespace(strengths=[], weaknesses=[], suggestions=[]),
        ),
    )
    generated_calls = []

    def generate(request, *, repetition):
        generated_calls.append(repetition)
        return generated

    namespace = {
        "json": json, "os": os, "tempfile": tempfile, "Path": Path,
        "datetime": datetime, "timezone": timezone, "GENERATION_CONFIG": config,
        "PROMPT_VERSION": prompt.version, "EXPERT_RUBRIC": SimpleNamespace(version=RUBRIC_ARTIFACT.version),
        "CONFIG_SHA256": "fake", "G2_RUN_DIRECTORY": tmp_path, "RATE_LIMIT_STATE": None,
        "ENABLE_LANGSMITH_TRACING": False, "tracing_context": lambda **kwargs: nullcontext(),
        "reserve_request_slot": lambda *args: None, "runner": SimpleNamespace(generate=generate),
        "PREPARED_DOCUMENTS": [{
            "already_saved": reused, "request": None, "repetition": 1,
            "document_id": "synthetic", "prediction_sha256": "fake", "comparison_sha256": "fake",
        }],
        "g2_runs": [SimpleNamespace(run_id="stale-from-previous-execution")],
    }
    exec(compile(generation, "g2-generation-tail", "exec"), namespace)
    assert namespace["g2_runs"] == ([] if reused else [generated])
    assert generated_calls == ([] if reused else [1])
    exec(compile(provenance, "g2-provenance", "exec"), namespace)
    artifact_path = tmp_path / "g2-run-synthetic.json"
    if reused:
        assert not artifact_path.exists()
    else:
        record = json.loads(artifact_path.read_text(encoding="utf-8"))["run"]
        assert record["prompt_artifact_sha256"] == prompt.sha256
        assert record["rendered_prompt_sha256"] == generated.rendered_prompt_sha256
        assert record["rubric_artifact_sha256"] == RUBRIC_ARTIFACT.sha256
        assert artifact_path.stat().st_mode & 0o777 == 0o600