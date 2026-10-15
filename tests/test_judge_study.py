import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import reflection_assessment_feedback.judge_study as study
from reflection_assessment_feedback.judge_stability import analyze_judge_study
from reflection_assessment_feedback.prompt_registry import render_prompt, resolve_prompt
from reflection_assessment_feedback.llm_judge import build_judge_input
from reflection_assessment_feedback.rubric_review import (
    create_rubric_review, rubric_review_cases, save_rubric_review_revision, validate_rubric_review,
)
from test_judge_batch import setup_batch
from test_selected_judge_analysis import save_results


@pytest.fixture
def frozen(setup_batch, monkeypatch):
    root, paths = save_results(setup_batch, monkeypatch)
    config = setup_batch[1]
    packets, _ = study.build_judge_batch(root, conditions=("G1", "G2", "G3"), pilot_only=False)
    by_id = {p["packet_id"]: p for p in packets}
    for path in paths:
        record = json.loads(path.read_text())
        prompt = resolve_prompt(record["prompt_id"], record["prompt_version"])
        _, digest = render_prompt(
            prompt, packet_json=json.dumps(
                build_judge_input(by_id[record["packet_id"]]), ensure_ascii=False, sort_keys=True, indent=2,
            ),
        )
        record.update({
            "rendered_prompt_sha256": digest, "judge_parameters": {},
            "run_id": f"old-{record['packet_id']}", "created_at": "2026-10-14T12:00:00+00:00",
        })
        path.write_text(json.dumps(record))
    monkeypatch.setattr(study, "load_experiment_config", lambda: config)
    directory = study.freeze_judge_study(root, cohort="playground")
    manifest = study.load_judge_study(directory)
    expected = {e["packet"]["packet_id"]: e for e in manifest["protocol"]["entries"]}

    def judge(**kwargs):
        entry = expected[kwargs["packet"]["packet_id"]]
        assert kwargs["packet"] == entry["packet"]
        record = copy.deepcopy(entry["imported_judgment"]["record"])
        record["run_id"] = f"new-{record['packet_id']}"
        return record

    calls = Mock(side_effect=judge)
    monkeypatch.setattr(study, "run_quality_judge", calls)
    monkeypatch.setattr(study, "reserve_request_slot", lambda *args: None)
    return directory, root, paths, calls, config


def execute(directory):
    return study.run_judge_study(
        directory, approve_external_processing=True, approve_full_batch=True,
        model=SimpleNamespace(temperature=None),
    )


def test_freeze_is_deterministic_private_and_preserves_sources(frozen):
    directory, root, paths, calls, _ = frozen
    original = {p: p.read_bytes() for p in paths}
    assert study.freeze_judge_study(root, cohort="playground") == directory
    assert study.study_progress(directory)["maximum_new_calls_without_retries"] == 24
    assert study.study_progress(directory)["saved_judgments"] == 12
    assert (directory / "manifest.json").stat().st_mode & 0o777 == 0o600
    assert directory.stat().st_mode & 0o777 == 0o700
    assert {p: p.read_bytes() for p in paths} == original
    calls.assert_not_called()


def test_frozen_execution_imports_resumes_and_is_independent_of_live_sources(frozen, monkeypatch):
    directory, root, paths, calls, config = frozen
    before = {p: p.read_bytes() for p in paths}
    # Neither changed prediction files nor changed generation settings reconstruct frozen inputs.
    for p in (root / "g2/predictions").glob("*.json"):
        p.write_text("{}")
    config["generation"]["expected_rubric_version"] = "changed"
    monkeypatch.setattr(study, "build_judge_batch", Mock(side_effect=AssertionError("Must not rebuild")))
    with pytest.raises(RuntimeError, match="approval"):
        study.run_judge_study(directory)
    result = execute(directory)
    assert result["saved_judgments"] == 36
    assert calls.call_count == 24
    execute(directory)
    assert calls.call_count == 24
    assert {p: p.read_bytes() for p in paths} == before
    frozen_study, saved = study.study_results(directory)
    entry = frozen_study["protocol"]["entries"][0]
    record = saved[(entry["packet"]["packet_id"], 1)]
    assert record["origin"] == "imported"
    assert record["judgment"]["created_at"] == "2026-10-14T12:00:00+00:00"
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in (directory / "results").glob("*.json"))


def test_failure_is_not_an_additional_repetition(frozen):
    directory, _, _, calls, _ = frozen
    generate = calls.side_effect
    calls.side_effect = RuntimeError("Provider failed")
    with pytest.raises(RuntimeError, match="Provider failed"):
        execute(directory)
    assert study.study_progress(directory)["saved_judgments"] == 12
    calls.side_effect = generate
    execute(directory)
    assert len(study.study_results(directory)[1]) == 36
    assert calls.call_count == 25


def test_mutated_manifest_and_results_fail(frozen):
    directory, _, _, _, _ = frozen
    execute(directory)
    path = next((directory / "results").glob("*.json"))
    original = path.read_text()
    duplicate = path.with_name("duplicate.json")
    duplicate.write_text(original)
    with pytest.raises(ValueError, match="Duplicate"):
        study.study_results(directory)
    duplicate.unlink()
    record = json.loads(original)
    record["frozen_input_sha256"] = "changed"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="frozen provenance"):
        study.study_results(directory)
    manifest = directory / "manifest.json"
    data = json.loads(manifest.read_text())
    data["protocol"]["judge_repetitions"] = 5
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="manifest hash"):
        study.load_judge_study(directory)


def test_ambiguous_imports_and_changed_judge_fail(frozen):
    directory, root, paths, _, config = frozen
    paths[0].with_name("judge-duplicate.json").write_bytes(paths[0].read_bytes())
    with pytest.raises(ValueError, match="Multiple import"):
        study.freeze_judge_study(root, cohort="playground")
    fresh = study.freeze_judge_study(root, cohort="playground", import_existing=False)
    assert study.study_progress(fresh)["eligible_imports"] == 0
    config["llm_judge"]["model_name"] = "changed-model"
    with pytest.raises(ValueError, match="configuration changed"):
        execute(directory)


def test_incompatible_import_parameters_fail(frozen):
    _, root, paths, _, _ = frozen
    record = json.loads(paths[0].read_text())
    record["judge_parameters"] = {"seed": 1}
    paths[0].write_text(json.dumps(record))
    with pytest.raises(ValueError, match="provenance mismatch"):
        study.freeze_judge_study(root, cohort="playground")


def test_concurrent_execution_and_wrong_model_are_rejected(frozen):
    import fcntl

    directory, _, _, calls, _ = frozen
    with (directory / ".study.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            execute(directory)
    with pytest.raises(ValueError, match="frozen judge model"):
        study.run_judge_study(
            directory, approve_external_processing=True, approve_full_batch=True,
            model=SimpleNamespace(model_name="wrong-model", temperature=None),
        )
    calls.assert_not_called()


def test_analysis_missingness_agreement_and_ordering(frozen):
    directory, _, _, _, _ = frozen
    tables = analyze_judge_study(directory)
    assert len(tables["coverage"]) == 36
    assert set(tables["coverage"].status) == {"missing", "imported"}
    assert len(tables["paired_ratings"]) == 10
    execute(directory)
    tables = analyze_judge_study(directory)
    assert len(tables["ratings"]) == 90
    assert set(tables["criterion_stability"].all_equal_proportion) == {1}
    assert set(tables["pairwise_agreement"].exact_agreement) == {1}
    assert len(tables["paired_medians"]) == 10
    assert set(tables["ordering_stability"].stable_ordering) == {True}
    assert set(tables["ordinal_counts"].direction) == {0}


def test_analysis_reversal_and_unable_to_judge(frozen):
    directory, _, _, _, _ = frozen
    execute(directory)
    manifest = study.load_judge_study(directory)
    entries = [
        e for e in manifest["protocol"]["entries"]
        if e["identity"]["document_id"] == "188" and e["packet"]["task"] == "assessment_quality"
    ]
    for entry in entries:
        if entry["identity"]["condition"] != "G1":
            continue
        for rep, score in [(2, 1), (3, 3)]:
            path = study._result_path(directory, entry["packet"]["packet_id"], rep)
            record = json.loads(path.read_text())
            record["judgment"]["result"]["ratings"][0]["score"] = score
            if rep == 3:
                record["judgment"]["result"]["ratings"][1].update(score=None, unable_to_judge=True)
            path.write_text(json.dumps(record))
    tables = analyze_judge_study(directory)
    assert tables["ordering_stability"].strict_reversal.eq(True).any()
    assert tables["ordering_stability"].tie_transition.eq(True).any()
    assert tables["criterion_stability"].extreme_1_to_3_packets.sum() == 1
    assert tables["criterion_stability"].unable_to_judge_ratings.sum() == 1
    assert len(tables["paired_medians"]) == 9


def test_review_requires_real_excerpts_clauses_and_evidence(frozen):
    directory = frozen[0]
    path = create_rubric_review(directory)
    worksheet = json.loads(path.read_text())
    cases = rubric_review_cases(directory)
    assert len(cases) == 18  # two documents, three conditions, three dimensions
    assert len(validate_rubric_review(directory, worksheet)) == 18
    row = worksheet["reviews"][0]
    row.update(status="reviewed", reviewer="Expert", notes="Reviewed against the frozen rubric.")
    revision = save_rubric_review_revision(directory, worksheet)
    assert save_rubric_review_revision(directory, worksheet) == revision
    row["findings"] = [{
        "issue": "theory_required", "classification": "ambiguous",
        "assessment_excerpt": "invented assessment", "rubric_clause": "invented rubric",
        "affects_score": None, "evidence_segment_ids": [], "rationale": "Review needed.",
    }]
    with pytest.raises(ValueError, match="excerpt"):
        save_rubric_review_revision(directory, worksheet)


def test_notebook_runs_local_paths_without_provider_calls(frozen, monkeypatch):
    directory, root, _, calls, _ = frozen
    monkeypatch.setenv("PREBI_DATA_ROOT", str(root))
    notebook = json.loads((Path(__file__).resolve().parents[1] / "11_judge_stability.ipynb").read_text())
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        exec(compile("".join(cell["source"]), f"11_judge_stability:{cell['id']}", "exec"), namespace)
        if cell["id"] == "study-setup":
            namespace["STUDY_DIRECTORY"] = directory
            assert not namespace["APPROVE_EXTERNAL_PROCESSING"]
            assert not namespace["APPROVE_FULL_BATCH"]
    assert namespace["tables"]["coverage"].shape[0] == 36
    calls.assert_not_called()


def test_revised_prompt_freezes_fresh_assessment_only_and_preserves_config(frozen, monkeypatch):
    _, root, paths, calls, config = frozen
    before = copy.deepcopy(config)
    originals = {p: p.read_bytes() for p in paths}
    directory = study.freeze_judge_study(
        root, cohort="playground", assessment_prompt_version="0.6.0",
        tasks=("assessment_quality",), import_existing=False,
    )
    manifest = study.load_judge_study(directory)
    entries = manifest["protocol"]["entries"]
    assert config == before
    assert len(entries) == 6
    assert study.study_progress(directory)["planned_judgments"] == 18
    assert study.study_progress(directory)["saved_judgments"] == 0
    assert all(e["expected_metadata"]["prompt_version"] == "0.6.0" for e in entries)
    assert all(e["imported_judgment"] is None for e in entries)
    assert all(e["packet"]["task"] == "assessment_quality" for e in entries)
    assert {p: p.read_bytes() for p in paths} == originals
    calls.assert_not_called()

    def judge(**kwargs):
        assert kwargs["assessment_prompt_version"] == "0.6.0"
        entry = next(e for e in entries if e["packet"]["packet_id"] == kwargs["packet"]["packet_id"])
        prompt = resolve_prompt("assessment_quality_llm_judge", "0.6.0")
        from test_llm_judge import FakeChatModel, _rating
        from reflection_assessment_feedback.llm_judge import run_quality_judge
        result = run_quality_judge(
            **{**kwargs, "model": FakeChatModel({
                "ratings": [_rating(c) for c in (
                    "score_rubric_fit", "evidence_support", "justification_quality",
                )],
                "overall_comment": "Synthetic.",
            }), "enable_langsmith_tracing": False},
        )
        assert result["rendered_prompt_sha256"] == entry["expected_metadata"]["rendered_prompt_sha256"]
        assert result["prompt_artifact_sha256"] == prompt.sha256
        return result

    calls.side_effect = judge
    execute(directory)
    assert calls.call_count == 18
    assert study.study_progress(directory)["saved_judgments"] == 18
    with pytest.raises(ValueError, match="fresh judgments"):
        study.freeze_judge_study(root, assessment_prompt_version="0.6.0")
    with pytest.raises(ValueError, match="assessment-quality tasks"):
        study.freeze_judge_study(
            root, assessment_prompt_version="0.6.0", tasks=("feedback_quality",),
            import_existing=False,
        )
