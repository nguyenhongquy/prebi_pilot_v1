import hashlib
import json
from pathlib import Path

import pytest

import reflection_assessment_feedback.selected_judge_analysis as analysis
from reflection_assessment_feedback.prompt_registry import resolve_prompt
from reflection_assessment_feedback.llm_judge import judge_input_sha256
from test_judge_batch import setup_batch, write_g2_fixture


def save_results(setup_batch, monkeypatch):
    root, config, _, _, _ = setup_batch
    write_g2_fixture(setup_batch)
    packets, key = analysis.build_judge_batch(root, conditions=("G1", "G2", "G3"), pilot_only=False)
    monkeypatch.setattr(analysis, "experiment_config_sha256", lambda: "config")
    monkeypatch.setattr(analysis, "load_experiment_config", lambda: config)
    notebook = json.loads((Path(__file__).resolve().parents[1] / "08_llm_judge_pilot.ipynb").read_text())
    source = next("".join(c["source"]) for c in notebook["cells"] if c["id"] == "4da92bb2")
    canonical_hash = lambda value: hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    namespace = {
        "key_by_packet": key, "JUDGE_MODE": "playground", "hashlib": hashlib,
        "judge_input_sha256": judge_input_sha256, "canonical_hash": canonical_hash,
        "JUDGE_PROVIDER": config["llm_judge"]["provider"],
        "JUDGE_MODEL_NAME": config["llm_judge"]["model_name"],
        "experiment_config_sha256": lambda: "config",
        "ASSESSMENT_PROMPT": resolve_prompt(config["prompts"]["assessment_quality_llm_id"], config["prompts"]["assessment_quality_llm_version"]),
        "FEEDBACK_PROMPT": resolve_prompt(config["prompts"]["feedback_quality_llm_id"], config["prompts"]["feedback_quality_llm_version"]),
    }
    exec(source[source.index("def expected_metadata("):source.index("def validate_saved_result(")], namespace)
    paths = []
    for packet in packets:
        condition = key[packet["packet_id"]]["condition"]
        directory = root / "llm-judge/development" / condition
        directory.mkdir(parents=True, exist_ok=True)
        task = packet["task"]
        criteria = (
            ("score_rubric_fit", "evidence_support", "justification_quality")
            if task == "assessment_quality" else ("correctness", "developmental_usefulness")
        )
        result = {
            "ratings": [{"criterion_id": c, "score": 2, "unable_to_judge": False,
                         "rationale": "Synthetic."} for c in criteria],
            "overall_comment": "Synthetic.",
        }
        if task == "feedback_quality":
            result["span_comments"] = []
        record = {**namespace["expected_metadata"](packet),
                  "record_type": "llm_judge_quality_rating", "result": result}
        path = directory / f"judge-{packet['packet_id']}.json"
        path.write_text(json.dumps(record))
        paths.append(path)
    return root, paths


def test_cross_batch_join_coverage_pairing_and_read_only(setup_batch, monkeypatch):
    root, paths = save_results(setup_batch, monkeypatch)
    before = {p: p.read_bytes() for p in root.rglob("*.json")}
    ratings, coverage, duplicates = analysis.load_selected_judge_results(root)
    assert len(coverage) == 12
    assert set(coverage.status) == {"complete"}
    assert duplicates.empty
    paired = analysis.paired_judge_comparison(ratings, ("G1", "G2", "G3"))
    assert len(paired) == 10
    assert {p: p.read_bytes() for p in root.rglob("*.json")} == before
    paths[0].unlink()
    ratings, coverage, _ = analysis.load_selected_judge_results(root)
    assert (coverage.status == "missing").sum() == 1
    assert len(analysis.paired_judge_comparison(ratings, ("G1", "G2", "G3"))) < 10


def test_duplicate_results_excluded_not_silently_selected(setup_batch, monkeypatch):
    root, paths = save_results(setup_batch, monkeypatch)
    copy = paths[0].parent / "judge-duplicate.json"
    copy.write_bytes(paths[0].read_bytes())
    ratings, coverage, duplicates = analysis.load_selected_judge_results(root)
    assert (coverage.status == "duplicate").sum() == 1
    assert len(duplicates) == 2
    assert duplicates.iloc[0].packet_id not in set(ratings.packet_id)


def test_provenance_mismatch_fails(setup_batch, monkeypatch):
    root, paths = save_results(setup_batch, monkeypatch)
    record = json.loads(paths[0].read_text())
    record["input_packet_sha256"] = "stale"
    paths[0].write_text(json.dumps(record))
    with pytest.raises(ValueError, match="provenance mismatch"):
        analysis.load_selected_judge_results(root)


def test_unable_to_judge_is_not_used_in_paired_comparison(setup_batch, monkeypatch):
    root, _ = save_results(setup_batch, monkeypatch)
    ratings, _, _ = analysis.load_selected_judge_results(root)
    ratings.loc[ratings.index[0], "unable_to_judge"] = True
    ratings.loc[ratings.index[0], "score"] = None
    assert len(analysis.paired_judge_comparison(ratings, ("G1", "G2", "G3"))) == 9


def test_notebook_executes_local_analysis_with_empty_coverage(setup_batch, monkeypatch):
    root = setup_batch[0]
    write_g2_fixture(setup_batch)
    monkeypatch.setenv("PREBI_DATA_ROOT", str(root))
    notebook = json.loads((Path(__file__).resolve().parents[1] / "09_llm_judge_analysis.ipynb").read_text())
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            assert "run_quality_judge(" not in source
            exec(compile(source, "judge-analysis-notebook", "exec"), namespace)
            if "ANALYSIS_COHORT =" in source:
                namespace["ANALYSIS_COHORT"] = "playground"
    assert namespace["ratings"].empty
    assert len(namespace["coverage"]) == 12
