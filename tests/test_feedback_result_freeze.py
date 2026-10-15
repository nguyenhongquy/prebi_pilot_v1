import json

import pandas as pd
import pytest

from reflection_assessment_feedback import feedback_result_freeze as freeze_module
from reflection_assessment_feedback.judge_study import _result_path

PACKETS = {
    ("a", "G1"): "00000000-0000-0000-0000-000000000001",
    ("a", "G3"): "00000000-0000-0000-0000-000000000003",
    ("b", "G2"): "00000000-0000-0000-0000-000000000002",
}


def run(condition, version="2.0.0"):
    components = [{"artifact_id": "generation", "version": version, "sha256": "p"}]
    if condition != "G1":
        components.append({"artifact_id": f"{condition.lower()}-supplement", "version": version, "sha256": "s"})
    return {
        "run_id": f"run-{condition}", "condition": condition, "prompt_version": version,
        "model_name": "gemini", "pipeline_version": "2.0.0", "rubric_version": "1.2.0",
        "rubric_artifact_sha256": "r", "rendered_prompt_sha256": f"rendered-{condition}",
        "prompt_components": components,
    }


def setup_studies(tmp_path, monkeypatch, *, g2_version="2.0.0", g2_inventory="inv", save_g2=True):
    root = tmp_path / "protected"
    studies = {}
    for study_id, conditions in (("a", ("G1", "G2", "G3")), ("b", ("G2",))):
        directory = root / "llm-judge" / "stability-studies" / study_id
        (directory / "results").mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps({"study_id": study_id}))
        sources, entries, results, rows = {}, [], {}, []
        for condition in conditions:
            identifier = PACKETS.get((study_id, condition), f"00000000-0000-0000-0000-00000000009{len(entries)}")
            version = g2_version if (study_id, condition) == ("b", "G2") else "2.0.0"
            source = json.dumps({"pairs": [{condition: run(condition, version)}]})
            digest = f"{study_id}-{condition}-source"
            sources[digest] = source
            entries.append({
                "identity": {"document_id": "7", "condition": condition, "run_id": f"run-{condition}",
                             "source_artifact_sha256": digest},
                "generation_repetition": 1, "input_sha256": f"input-{condition}",
                "packet": {"packet_id": identifier, "reference_inventory_ref": {
                    "inventory_id": g2_inventory if study_id == "b" else "inv",
                    "inventory_sha256": "inv-sha", "reflection_id": "7",
                }},
            })
            if save_g2 or study_id != "b":
                results[(identifier, 1)] = {"judgment": {}}
                _result_path(directory, identifier, 1).write_text(json.dumps({"condition": condition}))
            rows.append({
                "reflection_id": "7", "packet_id": identifier, "condition": condition,
                "generation_repetition": 1, "judge_repetition": 1,
                "positive_addressed_n": 1, "positive_rate_denominator_n": 2,
                "central_missing_n": 1, "central_partial_n": 1, "central_addressed_n": 2,
                "central_rate_denominator_n": 4, "any_major_issue": condition == "G2",
            })
        studies[directory] = ({"protocol": {
            "cohort": "remaining-test", "feedback_prompt_version": "0.12.0", "judge_repetitions": 1,
            "conditions": list(conditions), "judge_config": {"provider": "openai", "model_name": "judge"},
            "generation_sources": sources, "entries": entries,
        }}, results, pd.DataFrame(rows))
    monkeypatch.setattr(freeze_module, "study_results", lambda d: studies[d][:2])
    monkeypatch.setattr(
        freeze_module, "extract_feedback_measurements", lambda d: {"measurements": studies[d][2]},
    )
    return root


OPTIONS = {
    "label": "test", "selections": {"a": ("G1", "G3"), "b": ("G2",)},
    "generation_prompt_version": "2.0.0", "judge_prompt_version": "0.12.0",
}


def test_freeze_writes_once_verifies_and_detects_tampering(tmp_path, monkeypatch):
    root = setup_studies(tmp_path, monkeypatch)
    directory = freeze_module.freeze_feedback_results(root, **OPTIONS)
    assert freeze_module.freeze_feedback_results(root, **OPTIONS) == directory
    saved = freeze_module.verify_feedback_result_freeze(root, directory.name)
    record = saved["record"]
    assert {(i["generator"], i["study_id"]) for i in record["items"]} == {("G1", "a"), ("G2", "b"), ("G3", "a")}
    assert {i["generation"]["prompt_version"] for i in record["items"]} == {"2.0.0"}
    metrics = {m["Generator"]: m for m in record["metrics"]}
    assert (metrics["G2"]["CFCR"], metrics["G2"]["MDI"], metrics["G1"]["MDI"]) == (50.0, 100.0, 0.0)
    result = root / next(i["result_file"] for i in record["items"] if i["generator"] == "G2")
    result.write_text(json.dumps({"condition": "changed"}))
    with pytest.raises(ValueError, match="changed since freezing"):
        freeze_module.verify_feedback_result_freeze(root, directory.name)


def test_freeze_rejects_other_generation_prompt_versions(tmp_path, monkeypatch):
    root = setup_studies(tmp_path, monkeypatch, g2_version="1.4.0")
    with pytest.raises(ValueError, match="not generation prompt 2.0.0"):
        freeze_module.build_feedback_result_freeze(root, **OPTIONS)


@pytest.mark.parametrize("selections", [
    {"a": ("G1", "G3")}, {"a": ("G1", "G2", "G3"), "b": ("G2",)},
])
def test_freeze_requires_each_generator_once(tmp_path, monkeypatch, selections):
    root = setup_studies(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="exactly one study"):
        freeze_module.build_feedback_result_freeze(root, **{**OPTIONS, "selections": selections})


def test_freeze_rejects_inventory_mismatch_and_unsaved_judgments(tmp_path, monkeypatch):
    root = setup_studies(tmp_path / "inventory", monkeypatch, g2_inventory="other")
    with pytest.raises(ValueError, match="different inventories"):
        freeze_module.build_feedback_result_freeze(root, **OPTIONS)
    root = setup_studies(tmp_path / "unsaved", monkeypatch, save_g2=False)
    with pytest.raises(ValueError, match="unsaved"):
        freeze_module.build_feedback_result_freeze(root, **OPTIONS)
