import copy
import json
from unittest.mock import Mock

import pytest

from scripts import run_g2_feedback_evaluation as workflow
from reflection_assessment_feedback.judge_artifacts import content_hash
from reflection_assessment_feedback.reference_inventory import ExtractorConfig
from reflection_assessment_feedback.judge_study import load_judge_study
from test_judge_batch import setup_batch


def test_matching_inventory_selection_does_not_extract_or_approve(tmp_path, monkeypatch):
    config = workflow.load_experiment_config()
    source = {"synthetic": "reference-only"}
    extractor = ExtractorConfig(
        provider=config["llm_judge"]["provider"],
        model_name=config["llm_judge"]["model_name"], prompt_version="0.2.0",
    )
    monkeypatch.setattr(workflow, "inventory_specification", lambda source, config: {
        "source": source, "extractor": config.model_dump(mode="json"),
    })
    identifier = content_hash(workflow.inventory_specification(source, extractor))
    directory = tmp_path / "llm-judge" / "reference-inventories" / identifier
    directory.mkdir(parents=True)
    extract = Mock(side_effect=AssertionError("No provider calls during selection"))
    monkeypatch.setattr(workflow, "extract_reference_inventory", extract)

    references, pending = workflow.select_inventories(tmp_path, {"3": source}, config)
    assert references == {}
    assert pending["3"]["status"] == "missing"
    (directory / "draft.json").write_text("{}")
    record = {"inventory_id": identifier, "inventory_sha256": "hash", "inventory": {"reflection_id": "3"}}
    load = Mock(return_value=record)
    monkeypatch.setattr(workflow, "load_inventory_artifact", load)
    references, pending = workflow.select_inventories(tmp_path, {"3": source}, config)
    assert pending["3"]["status"] == "needs_review"
    (directory / "approval.json").write_text("{}")
    references, pending = workflow.select_inventories(tmp_path, {"3": source}, config)
    assert pending == {}
    assert references["3"]["inventory_id"] == identifier
    load.assert_called_with(directory, require_approved=True)
    extract.assert_not_called()


def test_freeze_requires_complete_inventory_coverage(tmp_path):
    with pytest.raises(ValueError, match="one matching inventory"):
        workflow.freeze_g2_feedback(tmp_path, {}, {}, tmp_path, {"document_ids": ["3"]}, {})


def test_freeze_new_g2_sources_is_stable_and_does_not_rebuild_legacy_batch(setup_batch, monkeypatch):
    root, config, comparisons, _, _ = setup_batch
    artifact = json.loads((comparisons / "comparison-188.json").read_text())
    run = copy.deepcopy(artifact["pairs"][0]["G1"])
    run["condition"] = "G2"
    run["run_id"] = "new-g2"
    run["document_id"] = "188"
    # Provenance/output validation is independently exercised by generator tests.
    monkeypatch.setattr(workflow, "_validate_run_provenance", Mock())
    monkeypatch.setattr(workflow, "validate_output", Mock())
    monkeypatch.setattr(workflow, "make_generation_request", Mock())
    monkeypatch.setattr(workflow.IntermediateAnalysis, "model_validate", Mock())
    from reflection_assessment_feedback.models import ReflectionDocument, ReflectionSegment
    import pandas as pd
    namespace = {
        "DEVELOPMENT_DOCUMENTS": {"188": ReflectionDocument(document_id="188", segments=[
            ReflectionSegment(segment_id="s", order=0, text="Synthetic reflection."),
        ])},
        "COHORT_RECORDS": pd.DataFrame([{"document_id": "188", "feedback_text": "Synthetic reference."}]),
    }
    batch = root / "new-g2-batch"
    (batch / "predictions").mkdir(parents=True)
    (batch / "generation").mkdir()
    prediction = {"analysis": {}}
    (batch / "predictions" / "document-188.json").write_text(json.dumps(prediction))
    provenance = {"document_ids": ["188"]}
    envelope = {
        "document_id": "188", "prediction_sha256": workflow.fingerprint(prediction),
        "provenance": provenance, "run": run,
    }
    (batch / "generation" / "document-188-rep-1.json").write_text(json.dumps(envelope))

    # Inspect the real seed through load_judge_study; avoid approval/provider behavior.
    def freeze(root, **kwargs):
        seed = load_judge_study(kwargs["source_study"])
        assert kwargs["conditions"] == ("G2",)
        assert kwargs["feedback_prompt_version"] == "0.12.0"
        assert kwargs["judge_repetitions"] == 1
        assert len(seed["protocol"]["entries"]) == 1
        assert seed["protocol"]["entries"][0]["identity"]["run_id"] == "new-g2"
        assert seed["protocol"]["entries"][0]["identity"]["cohort_id"] == config["dataset"]["generation_cohort"]
        assert seed["protocol"]["entries"][0]["identity"]["task"] == "feedback_quality"
        return kwargs["source_study"]

    monkeypatch.setattr(workflow, "freeze_judge_study", freeze)
    first = workflow.freeze_g2_feedback(root, namespace, config, batch, provenance, {"188": {}})
    second = workflow.freeze_g2_feedback(root, namespace, config, batch, provenance, {"188": {}})
    assert first == second
