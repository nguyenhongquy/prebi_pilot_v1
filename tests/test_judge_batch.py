import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

import reflection_assessment_feedback.judge_batch as batch
from reflection_assessment_feedback import EXPERT_RUBRIC
from reflection_assessment_feedback.models import (
    HumanAnalysisCandidate, HumanAnalysisContext, HumanSegmentAnalysis,
    ReflectionDocument, ReflectionSegment,
)
from reflection_assessment_feedback.llm_judge import build_judge_input
from reflection_assessment_feedback.prompt import build_prompt, prompt_components_for_request
from reflection_assessment_feedback import make_generation_request


@pytest.fixture
def setup_batch(tmp_path, monkeypatch):
    config = batch.load_experiment_config()
    monkeypatch.setattr(batch, "load_experiment_config", lambda: config)
    monkeypatch.setattr(batch, "experiment_config_sha256", lambda: "config")
    rubric = batch.resolve_rubric(config["generation"]["rubric_id"], config["generation"]["expected_rubric_version"])
    feedback = pd.DataFrame([
        {"document_id": str(i), "feedback_text": f"Synthetic reference {i}"}
        for i in (188, 126)
    ])
    monkeypatch.setattr(batch, "load_playground_feedback_cohort", lambda: feedback)
    remaining = Mock(return_value=feedback)
    monkeypatch.setattr(batch, "load_remaining_test_feedback_cohort", remaining)
    human = HumanAnalysisContext(segments=[HumanSegmentAnalysis(segment_id="s", candidates=[
        HumanAnalysisCandidate(
            candidate_id="c", scope="in_scope", scope_known=True, raw_scope_value="1",
            component_bands={"SW": "1", "UA": "0", "HA": "0"},
            raw_component_values={"SW": "1", "UA": "0", "HA": "0"},
            known_component_bands={"SW": True, "UA": True, "HA": True}, requires_review=False,
        )
    ])])
    documents = {str(i): ReflectionDocument(document_id=str(i), segments=[
        ReflectionSegment(segment_id="s", text="Synthetic reflection.", order=0)
    ]) for i in (188, 126)}
    monkeypatch.setattr(batch, "load_human_analysis_context", lambda path, identifier: (documents[identifier], human))
    summary = {"analysis_source": "provisional_gold_candidate_annotations", "segment_count": 1}
    monkeypatch.setattr(batch, "build_gold_segment_analysis_summary", lambda *args: summary)
    g2 = Mock(wraps=batch.IntermediateAnalysis.model_validate)
    monkeypatch.setattr(batch.IntermediateAnalysis, "model_validate", g2)
    claim = {"interpretation": "Synthetic interpretation.", "evidence_refs": ["s"]}
    output = {
        "evidence_synthesis": [
            {"dimension_id": dimension, **{field: [claim] for field in (
                "demonstrated_depth", "development", "integrative_coherence",
            )}, "limitations": [], "uncertainty": []} for dimension in ("SW", "UA", "HA")
        ],
        "assessment": {"component_assessments": [
            {"dimension_id": d, "band": 1, "justification": "Synthetic justification.",
             "evidence_segment_ids": ["s"], "intermediate_score_explanation": None}
            for d in ("SW", "UA", "HA")
        ]},
        "feedback": {"strengths": [{"message": "Synthetic strength.", "evidence_segment_ids": ["s"]}], "weaknesses": [], "suggestions": []},
    }
    directory = tmp_path / "g1-g3" / "demo-runs"
    directory.mkdir(parents=True)
    for identifier in documents:
        runs = {}
        for condition in ("G1", "G3"):
            request = make_generation_request(condition=condition, document=documents[identifier], rubric=EXPERT_RUBRIC, analysis=human if condition == "G3" else None)
            components = prompt_components_for_request(request)
            runs[condition] = {
                "run_id": f"{condition}-{identifier}", "condition": condition, "repetition": 1,
                "model_name": config["generation"]["model_name"], "generation_parameters": {},
                "prompt_version": config["generation"]["expected_prompt_version"],
                "prompt_artifact_sha256": components[0].sha256,
                "rendered_prompt_sha256": hashlib.sha256(build_prompt(request).encode()).hexdigest(),
                "prompt_components": [{"artifact_id": a.artifact_id, "version": a.version, "sha256": a.sha256} for a in components],
                "rubric_version": "1.2.0", "pipeline_version": "2.0.0",
                "rubric_artifact_sha256": rubric.sha256, "output": copy.deepcopy(output),
            }
        artifact = {
            "document_id": identifier, "development_cohort": "playground-author-188-v1",
            "experiment_config_sha256": "config", "rubric_version": "1.2.0",
            "rubric_artifact_sha256": rubric.sha256,
            "g3_human_analysis_context": human.model_dump(mode="json"),
            "pairs": [{"repetition": 1, **runs}],
        }
        (directory / f"comparison-{identifier}.json").write_text(json.dumps(artifact))
    return tmp_path, config, directory, g2, remaining


def test_playground_g1_g3_needs_no_g2_or_packet_export_and_ids_are_stable(setup_batch):
    root, _, _, g2, remaining = setup_batch
    packets, key = batch.build_judge_batch(root)
    assert len(packets) == 4
    assert {row["condition"] for row in key.values()} == {"G1", "G3"}
    assert {row["document_id"] for row in key.values()} == {"188"}
    assert batch.build_judge_batch(root) == (packets, key)
    for packet in packets:
        inputs = build_judge_input(packet)
        assert not {"condition", "run_id", "model_name", "evidence_synthesis"} & inputs.keys()
        assert inputs["rubric"]["version"] == "1.2.0"
    assert len(list(root.rglob("*.json"))) == 2
    g2.assert_not_called()
    remaining.assert_not_called()


def test_full_cohort_and_condition_task_selection(setup_batch):
    root, _, _, _, _ = setup_batch
    packets, key = batch.build_judge_batch(root, conditions=("G3",), tasks=("feedback_quality",), pilot_only=False)
    assert len(packets) == 2
    assert {row["document_id"] for row in key.values()} == {"126", "188"}
    assert all(row["condition"] == "G3" for row in key.values())
    assert all("gold_analysis_summary" not in p for p in packets)


@pytest.mark.parametrize("options", [
    {"cohort": "unknown"}, {"conditions": ()}, {"conditions": ("G1", "G1")},
    {"conditions": ("G4",)}, {"tasks": ("unknown",)},
])
def test_invalid_selections_fail(setup_batch, options):
    with pytest.raises(ValueError):
        batch.build_judge_batch(setup_batch[0], **options)


def test_duplicates_missing_and_stale_artifacts_fail(setup_batch):
    root, _, directory, _, _ = setup_batch
    path = directory / "comparison-188.json"
    original = path.read_text()
    duplicate = directory / "comparison-duplicate.json"
    duplicate.write_text(original)
    with pytest.raises(ValueError, match="one compatible"):
        batch.build_judge_batch(root)
    duplicate.unlink()
    record = json.loads(original)
    record["experiment_config_sha256"] = "historical"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="one compatible"):
        batch.build_judge_batch(root)


def test_remaining_test_selection_uses_its_cohort_and_count(setup_batch):
    root, config, directory, _, remaining = setup_batch
    config["dataset"]["expected_generation_document_count"] = 2
    for path in directory.glob("*.json"):
        record = json.loads(path.read_text())
        record["development_cohort"] = config["dataset"]["generation_cohort"]
        path.write_text(json.dumps(record))
    packets, _ = batch.build_judge_batch(root, cohort="remaining-test", pilot_only=False)
    assert len(packets) == 8
    remaining.assert_called_once()


def test_saved_prompt_snapshot_must_match_original_reflection(setup_batch):
    root, _, directory, _, _ = setup_batch
    path = directory / "comparison-188.json"
    record = json.loads(path.read_text())
    record["pairs"][0]["G1"]["rendered_prompt_sha256"] = "different-reflection"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="reflection snapshot"):
        batch.build_judge_batch(root)


def write_g2_fixture(setup_batch, repetitions=1):
    root, config, directory, _, _ = setup_batch
    config["generation"]["repetitions"] = repetitions
    prediction_dir = root / "g2" / "predictions"
    run_dir = root / "g2" / "demo-runs"
    prediction_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    for identifier in ("188", "126"):
        path = directory / f"comparison-{identifier}.json"
        prediction = {
            "document_id": identifier, "experiment_config_sha256": "config",
            "analysis": {"source": "predicted", "segments": [
                {"segment_id": "s", "scope": "in_scope",
                 "component_bands": {"SW": "1", "UA": "0", "HA": "0"}}
            ]},
        }
        prediction_path = prediction_dir / f"predicted-analysis-{identifier}.json"
        prediction_path.write_text(json.dumps(prediction))
        document = ReflectionDocument(document_id=identifier, segments=[
            ReflectionSegment(segment_id="s", text="Synthetic reflection.", order=0),
        ])
        request = make_generation_request(
            condition="G2", document=document, rubric=EXPERT_RUBRIC,
            analysis=batch.IntermediateAnalysis.model_validate(prediction["analysis"]),
        )
        run = json.loads(path.read_text())["pairs"][0]["G1"]
        run.update({
            "condition": "G2", "document_id": identifier, "analysis_source": "predicted",
            "analysis_reference_id": f"predicted-analysis-sha256-{hashlib.sha256(prediction_path.read_bytes()).hexdigest()}",
            "rendered_prompt_sha256": hashlib.sha256(build_prompt(request).encode()).hexdigest(),
            "prompt_components": [
                {"artifact_id": a.artifact_id, "version": a.version, "sha256": a.sha256}
                for a in prompt_components_for_request(request)
            ],
        })
        for repetition in range(1, repetitions + 1):
            run["repetition"] = repetition
            (run_dir / f"g2-run-{identifier}-{repetition}.json").write_text(json.dumps({
                "record_type": "g2_generation_run", "experiment_config_sha256": "config",
                "paired_g1_g3_comparison_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "run": run,
            }))
    return run_dir


@pytest.mark.parametrize("repetitions", [1, 2])
def test_g2_notebook_artifacts_support_pilot_full_cohort_and_repetitions(setup_batch, repetitions):
    root = setup_batch[0]
    write_g2_fixture(setup_batch, repetitions)
    packets, key = batch.build_judge_batch(root, conditions=("G2",))
    assert len(packets) == 2 * repetitions
    assert {row["document_id"] for row in key.values()} == {"188"}
    assert batch.build_judge_batch(root, conditions=("G2",)) == (packets, key)
    packets, key = batch.build_judge_batch(root, conditions=("G2",), pilot_only=False)
    assert len(packets) == 4 * repetitions
    assert {row["document_id"] for row in key.values()} == {"188", "126"}


@pytest.mark.parametrize("mutation", ["prediction", "paired", "duplicate", "prompt"])
def test_g2_stale_or_ambiguous_notebook_inputs_rejected(setup_batch, mutation):
    root = setup_batch[0]
    directory = write_g2_fixture(setup_batch)
    path = directory / "g2-run-188-1.json"
    record = json.loads(path.read_text())
    if mutation == "prediction":
        record["run"]["analysis_reference_id"] = "stale"
    elif mutation == "paired":
        record["paired_g1_g3_comparison_sha256"] = "stale"
    elif mutation == "duplicate":
        (directory / "g2-run-duplicate.json").write_text(path.read_text())
    else:
        record["run"]["rendered_prompt_sha256"] = "stale"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="compatible notebook G2|reflection snapshot"):
        batch.build_judge_batch(root, conditions=("G2",))


def test_notebook_executes_only_selected_packets_and_resumes(setup_batch, monkeypatch):
    import langchain_openai
    import reflection_assessment_feedback.llm_judge as judge
    import reflection_assessment_feedback.rate_limit as rate_limit

    root, _, _, _, _ = setup_batch
    packets, key = batch.build_judge_batch(root)
    notebook = json.loads((Path(__file__).resolve().parents[1] / "08_llm_judge_pilot.ipynb").read_text())
    cells = {c["id"]: "".join(c["source"]) for c in notebook["cells"]}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    config = batch.load_experiment_config()
    prompts = config["prompts"]
    from reflection_assessment_feedback.prompt_registry import resolve_prompt
    namespace = {
        "APPROVE_EXTERNAL_PROCESSING": True, "PILOT_ONLY": True, "APPROVE_FULL_BATCH": False,
        "os": __import__("os"), "Path": Path, "UUID": __import__("uuid").UUID, "json": json,
        "Counter": __import__("collections").Counter,
        "JUDGE_PROVIDER": "openai", "JUDGE_MODEL_NAME": "fake", "JUDGE_MODE": "playground",
        "JUDGE_COHORT": "playground", "CONDITIONS_TO_JUDGE": ("G1", "G3"),
        "TASKS_TO_JUDGE": ("assessment_quality", "feedback_quality"),
        "ENABLE_LANGSMITH_TRACING": False, "quality_packets": packets, "key_by_packet": key,
        "PROTECTED_ROOT": root, "JUDGE_CONFIG": {"minimum_request_interval_seconds": 1},
        "JUDGE_DOCUMENT_IDS": {"188"},
        "BATCH_SELECTION": ("playground", ("G1", "G3"), ("assessment_quality", "feedback_quality"), True),
        "ASSESSMENT_PROMPT": resolve_prompt(prompts["assessment_quality_llm_id"], prompts["assessment_quality_llm_version"]),
        "FEEDBACK_PROMPT": resolve_prompt(prompts["feedback_quality_llm_id"], prompts["feedback_quality_llm_version"]),
    }
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key")
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **kwargs: object())
    monkeypatch.setattr(rate_limit, "reserve_request_slot", lambda *args: None)

    def fake_judge(**kwargs):
        packet = kwargs["packet"]
        criteria = ("score_rubric_fit", "evidence_support", "justification_quality") if packet["task"] == "assessment_quality" else ("correctness", "developmental_usefulness")
        output = {"ratings": [{"criterion_id": c, "score": 2, "unable_to_judge": False, "rationale": "Synthetic."} for c in criteria], "overall_comment": "Synthetic."}
        if packet["task"] == "feedback_quality":
            output["span_comments"] = []
        return {"record_type": "llm_judge_quality_rating", "result": output}

    calls = Mock(side_effect=fake_judge)
    monkeypatch.setattr(judge, "run_quality_judge", calls)
    exec(cells["4da92bb2"], namespace)
    assert calls.call_count == 4
    exec(cells["4da92bb2"], namespace)
    assert calls.call_count == 4
    exec(cells["b22679f5"], namespace)
    assert len(namespace["saved_results"]) == 4
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in namespace["RESULT_DIRECTORY"].glob("judge-*.json"))
    assert "run_quality_judge(" not in cells["5a20fdce"]
    result_path = next(namespace["RESULT_DIRECTORY"].glob("judge-*.json"))
    saved = json.loads(result_path.read_text())
    saved["input_packet_sha256"] = "stale"
    result_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="provenance"):
        exec(cells["4da92bb2"], namespace)
    assert calls.call_count == 4
    namespace["CONDITIONS_TO_JUDGE"] = ("G1",)
    with pytest.raises(RuntimeError, match="Selection changed"):
        exec(cells["4da92bb2"], namespace)
    namespace["CONDITIONS_TO_JUDGE"] = ("G1", "G3")
    namespace["APPROVE_EXTERNAL_PROCESSING"] = False
    with pytest.raises(RuntimeError, match="approval"):
        exec(cells["4da92bb2"], namespace)
