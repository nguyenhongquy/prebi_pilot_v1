import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import pandas as pd
from pydantic import ValidationError

import reflection_assessment_feedback.reference_inventory as inventories
import reflection_assessment_feedback.judge_study as studies
import reflection_assessment_feedback.judge_stability as stability
from reflection_assessment_feedback.judge_artifacts import content_hash
from reflection_assessment_feedback.judge_stability import analyze_judge_study
from reflection_assessment_feedback.prompt_registry import resolve_prompt
from reflection_assessment_feedback.llm_judge import (
    ApplicableConcernJudgeOutput, EvaluableConcernJudgeOutput, FrozenConcernJudgeOutput,
    ReferencePriorityJudgeOutput,
    RefinedReferencePriorityJudgeOutput,
    build_judge_input, run_quality_judge,
    validate_frozen_concern_output,
)
from reflection_assessment_feedback.reference_inventory import (
    ExtractorConfig, ReferenceConcernInventory, approve_inventory,
    extract_reference_inventory, inventory_reference, inventory_specification,
    load_inventory_artifact, reference_input_from_packet, validate_inventory,
    EvaluableConcernInventory,
)
from test_feedback_alignment import packet
from test_judge_batch import setup_batch
from test_judge_study import frozen
from test_llm_judge import FakeChatModel


def inventory_data(source):
    return {
        "schema_version": "1.0.0", "reflection_id": source["reflection_id"],
        "reference_concerns": [{
            "concern_id": "R1", "concern_type": "developmental",
            "human_excerpt": source["reference_teacher_feedback"],
            "normalized_concern": "Explain the reasons.",
            "centrality": "central", "centrality_rationale": "Main development priority.",
            "reference_support": "supported", "evidence_segment_ids": ["s1"],
            "rationale": "Documented lecturer priority.",
        }],
    }


def draft(root, source, monkeypatch):
    model = FakeChatModel(inventory_data(source))
    model.temperature = None
    monkeypatch.setattr(inventories, "reserve_request_slot", lambda *args: None)
    directory = extract_reference_inventory(
        root, source=source, config=ExtractorConfig(provider="test", model_name="fake"),
        model=model, approve_external_processing=True,
    )
    return directory, model


def frozen_output(coverage=None):
    return {
        "schema_version": "2.0.0",
        "concern_coverage": coverage if coverage is not None else [{
            "concern_id": "R1", "coverage": "missing", "generated_references": [],
            "evidence_segment_ids": [], "rationale": "Not covered.",
        }],
        "issues": [], "overall_severity": "no_error",
        "rationale": "Missing coverage is not automatically a substantive issue.",
        "reference_uncertainties": [],
    }


def with_inventory(p, data):
    p = copy.deepcopy(p)
    p["reference_inventory"] = data
    p["reference_inventory_ref"] = {
        "inventory_id": "a" * 64, "inventory_sha256": content_hash(data),
        "reflection_id": data["reflection_id"],
    }
    return p


def test_reference_only_input_cache_and_explicit_approval(tmp_path, monkeypatch):
    p = packet()
    source = reference_input_from_packet(p, "3")
    directory, model = draft(tmp_path, source, monkeypatch)
    payload = json.loads(model.runnable.prompt.split("PAKET:\n")[1])
    assert set(payload) == {
        "reflection_id", "reflection", "reference_teacher_feedback", "rubric", "rubric_version",
    }
    assert not {"output", "condition", "gold_analysis_summary"} & payload.keys()
    before = (directory / "draft.json").read_bytes()
    with pytest.raises(ValueError, match="explicit approval"):
        load_inventory_artifact(directory, require_approved=True)
    # No model, approval or provider call is needed for a cached draft.
    assert extract_reference_inventory(
        tmp_path, source=source, config=ExtractorConfig(provider="test", model_name="fake"),
    ) == directory
    approve_inventory(directory, approved_by="test-reviewer")
    approve_inventory(directory, approved_by="test-reviewer")
    record = load_inventory_artifact(directory, require_approved=True)
    assert inventory_reference(record)["inventory_sha256"] == content_hash(record["inventory"])
    assert (directory / "draft.json").read_bytes() == before
    assert (directory / "draft.json").stat().st_mode & 0o777 == 0o600
    assert directory.stat().st_mode & 0o777 == 0o700
    changed = json.loads(before)
    changed["inventory"]["reference_concerns"][0]["normalized_concern"] = "Tampered"
    (directory / "draft.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="content hash"):
        load_inventory_artifact(directory, require_approved=True)
    changed["inventory_sha256"] = content_hash(changed["inventory"])
    (directory / "draft.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="finalized artifact hash"):
        load_inventory_artifact(directory, require_approved=True)


@pytest.mark.parametrize("field", ["reflection", "reference_teacher_feedback", "rubric", "rubric_version"])
def test_source_changes_invalidate_cache(field):
    source = reference_input_from_packet(packet(), "3")
    config = ExtractorConfig(provider="test", model_name="fake")
    original = content_hash(inventory_specification(source, config))
    source[field] = {"changed": True} if isinstance(source[field], dict) else "changed"
    assert content_hash(inventory_specification(source, config)) != original


def test_extractor_version_and_configuration_invalidate_cache(monkeypatch):
    source = reference_input_from_packet(packet(), "3")
    config = ExtractorConfig(provider="test", model_name="fake")
    original = content_hash(inventory_specification(source, config))
    for update in (
        {"parameters": {"temperature": 0}}, {"model_name": "other"}, {"extraction_revision": "2"},
    ):
        assert content_hash(inventory_specification(source, config.model_copy(update=update))) != original
    resolve = inventories.resolve_prompt
    monkeypatch.setattr(inventories, "resolve_prompt", lambda identity, version: resolve(identity, "0.1.0"))
    assert content_hash(inventory_specification(
        source, config.model_copy(update={"prompt_version": "0.2.0"}),
    )) != original


@pytest.mark.parametrize("change", ["quote", "segment", "id", "order"])
def test_inventory_validation(change):
    source = reference_input_from_packet(packet(), "3")
    data = inventory_data(source)
    if change == "quote":
        data["reference_concerns"][0]["human_excerpt"] = "Invented"
    elif change == "segment":
        data["reference_concerns"][0]["evidence_segment_ids"] = ["unknown"]
    elif change == "id":
        data["reference_concerns"][0]["concern_id"] = "R2"
    else:
        source["reference_teacher_feedback"] = "First. Second."
        data["reference_concerns"][0]["human_excerpt"] = "Second."
        extra = copy.deepcopy(data["reference_concerns"][0])
        extra.update(concern_id="R2", human_excerpt="First.")
        data["reference_concerns"].append(extra)
    with pytest.raises(ValueError):
        validate_inventory(ReferenceConcernInventory.model_validate(data), source)


@pytest.mark.parametrize("change", ["delete", "introduce", "duplicate", "modify"])
def test_coverage_cannot_change_inventory(change):
    source = reference_input_from_packet(packet(), "3")
    p = with_inventory(packet(), inventory_data(source))
    data = frozen_output()
    if change == "delete":
        data["concern_coverage"] = []
    elif change == "introduce":
        data["concern_coverage"][0]["concern_id"] = "R2"
    elif change == "duplicate":
        data["concern_coverage"].append(copy.deepcopy(data["concern_coverage"][0]))
    else:
        data["concern_coverage"][0]["normalized_concern"] = "Changed"
    with pytest.raises(ValueError):
        validate_frozen_concern_output(FrozenConcernJudgeOutput.model_validate(data), p)


def test_missing_and_uncertain_concerns_are_not_automatic_issues():
    source = reference_input_from_packet(packet(), "3")
    data = inventory_data(source)
    data["reference_concerns"][0]["reference_support"] = "uncertain"
    p = with_inventory(packet(), data)
    validate_frozen_concern_output(FrozenConcernJudgeOutput.model_validate(frozen_output()), p)
    model = FakeChatModel(frozen_output())
    result = run_quality_judge(
        model=model, packet=p, provider="test", model_name="fake", feedback_prompt_version="0.8.0",
    )
    assert result["schema_version"] == "2.0.0"
    assert result["reference_inventory_ref"] == p["reference_inventory_ref"]
    assert "reference_concerns" not in result["result"]
    payload = json.loads(model.runnable.prompt.split("PAKET:\n")[1])
    assert payload["reference_inventory"] == data


def test_versions_do_not_confuse_feedback_with_blind_assessment():
    source = reference_input_from_packet(packet(), "3")
    p = with_inventory(packet(), inventory_data(source))
    assert "reference_teacher_feedback" in build_judge_input(p, feedback_prompt_version="0.8.0")
    with pytest.raises(ValueError, match="approved reference inventory"):
        build_judge_input(packet(), feedback_prompt_version="0.8.0")


@pytest.mark.parametrize("first_coverage", ["addressed", "unable_to_judge"])
@pytest.mark.parametrize("version", ["0.8.0", "0.9.0", "0.10.0", "0.11.0", "0.12.0"])
def test_conditions_and_repetitions_reuse_approved_inventory(frozen, monkeypatch, first_coverage, version):
    original, root, _, calls, _ = frozen
    entries = studies.load_judge_study(original)["protocol"]["entries"]
    references = {}
    extraction_calls = Mock(wraps=inventories.invoke_judge_model)
    monkeypatch.setattr(inventories, "invoke_judge_model", extraction_calls)
    for entry in entries:
        document = entry["identity"]["document_id"]
        if document in references or entry["packet"]["task"] != "feedback_quality":
            continue
        source = reference_input_from_packet(entry["packet"], document)
        data = inventory_data(source)
        data["reference_concerns"][0]["evidence_segment_ids"] = []
        # A second central developmental concern and a positive one exercise denominators.
        for index, kind in ((2, "developmental"), (3, "positive")):
            extra = copy.deepcopy(data["reference_concerns"][0])
            extra.update(concern_id=f"R{index}", concern_type=kind)
            data["reference_concerns"].append(extra)
        if version in {"0.9.0", "0.10.0", "0.11.0", "0.12.0"}:
            data["schema_version"] = "1.1.0"
            for concern in data["reference_concerns"]:
                concern.update(
                    evaluability="evaluable", evaluability_rationale="Current text.",
                    evaluable_scope="Current reasons.", unavailable_context=[],
                )
            data["reference_concerns"][0].update(
                evaluability="partially_evaluable", unavailable_context=["Earlier version"],
            )
            excluded = copy.deepcopy(data["reference_concerns"][0])
            excluded.update(concern_id="R4", evaluability="not_evaluable", evaluable_scope=None)
            data["reference_concerns"].append(excluded)
        model = FakeChatModel(data)
        model.temperature = None
        directory = extract_reference_inventory(
            root, source=source, config=ExtractorConfig(
                provider="test", model_name="fake",
                prompt_version="0.2.0" if version in {"0.9.0", "0.10.0", "0.11.0", "0.12.0"} else "0.1.0",
            ),
            model=model, approve_external_processing=True,
        )
        with pytest.raises(ValueError, match="explicit approval"):
            load_inventory_artifact(directory, require_approved=True)
        with pytest.raises(ValueError, match="explicit approval"):
            studies.freeze_judge_study(
                root, cohort="playground", tasks=("feedback_quality",), import_existing=False,
                feedback_prompt_version=version, source_study=original,
                reference_inventory_refs={
                    document: inventory_reference(load_inventory_artifact(directory)),
                }, document_ids=(document,),
            )
        references[document] = approve_inventory(directory, approved_by="test-reviewer")
    assert extraction_calls.call_count == len(references)
    directory = studies.freeze_judge_study(
        root, cohort="playground", tasks=("feedback_quality",), import_existing=False,
        feedback_prompt_version=version, source_study=original, judge_repetitions=2,
        reference_inventory_refs=references,
    )
    manifest = studies.load_judge_study(directory)
    assert len(manifest["protocol"]["reference_inventories"]) == len(references)
    by_document = {}
    for entry in manifest["protocol"]["entries"]:
        document = entry["identity"]["document_id"]
        assert "reference_inventory" not in entry["packet"]
        assert entry["packet"]["reference_inventory_ref"] == references[document]
        inventory = studies.evaluation_packet(manifest["protocol"], entry)["reference_inventory"]
        assert by_document.setdefault(document, inventory) == inventory

    def judge(**kwargs):
        p = kwargs["packet"]
        data = frozen_output([
            {"concern_id": c["concern_id"], "coverage": "unable_to_judge",
             "generated_references": [], "evidence_segment_ids": [], "rationale": "Local uncertainty."}
            for c in p["reference_inventory"]["reference_concerns"]
            if c.get("evaluability") != "not_evaluable"
        ])
        if version == "0.9.0":
            data["schema_version"] = "2.1.0"
        if version in {"0.10.0", "0.11.0", "0.12.0"}:
            data["schema_version"] = "2.3.0" if version in {"0.11.0", "0.12.0"} else "2.2.0"
            data["concern_coverage"].append({
                "concern_id": "R4", "coverage": "not_applicable",
                "generated_references": [], "evidence_segment_ids": [],
                "rationale": "Earlier version was unavailable.",
            })
        if first_coverage == "addressed":
            data["concern_coverage"][0].update(
                coverage="addressed", generated_references=[{
                    "component": "strengths", "item_index": 0,
                    "excerpt": p["output"]["strengths"][0]["text"],
                }],
            )
        return run_quality_judge(**{
            **kwargs, "model": FakeChatModel(data), "enable_langsmith_tracing": False,
        })

    calls.side_effect = judge
    for _ in range(2):
        studies.run_judge_study(
            directory, approve_external_processing=True, approve_full_batch=True,
            model=SimpleNamespace(temperature=None),
        )
    assert calls.call_count == len(entries)  # Two tasks originally; now feedback x two repetitions.
    assert extraction_calls.call_count == len(references)
    tables = analyze_judge_study(directory)
    assert set(tables["decisions"].central_developmental_count) == {3 if version in {"0.9.0", "0.10.0", "0.11.0", "0.12.0"} else 2}
    assert set(tables["decisions"].central_developmental_not_evaluable) == {1 if version in {"0.9.0", "0.10.0", "0.11.0", "0.12.0"} else 0}
    if version in {"0.9.0", "0.10.0", "0.11.0", "0.12.0"}:
        assert set(tables["decisions"].central_developmental_partially_evaluable) == {1}
        assert set(tables["concern_coverage"].loc[
            tables["concern_coverage"].concern_id == "R4", "coverage"
        ]) == {"not_applicable" if version in {"0.10.0", "0.11.0", "0.12.0"} else "excluded_not_evaluable"}
    if version in {"0.10.0", "0.11.0", "0.12.0"}:
        assert set(tables["decisions"].not_applicable_count) == {1}
        assert set(tables["decisions"].central_developmental_not_applicable) == {1}
    if first_coverage == "addressed":
        assert set(tables["decisions"].central_developmental_unable_to_judge) == {1}
        assert set(tables["decisions"].central_developmental_evaluable) == {1}
        assert set(tables["decisions"].central_developmental_addressed_proportion) == {1}
    else:
        assert set(tables["decisions"].central_developmental_unable_to_judge) == {2}
        assert set(tables["decisions"].central_developmental_evaluable) == {0}
        assert tables["decisions"].central_developmental_addressed_proportion.isna().all()
    assert set(tables["decisions"].major_issues) == {0}
    assert set(tables["concern_coverage"].concern_type) == {"positive", "developmental"}
    measurements = tables["measurements"]
    assert list(measurements) == stability.FEEDBACK_MEASUREMENT_COLUMNS
    assert len(measurements) == len(entries)
    assert not measurements.duplicated(["packet_id", "judge_repetition"]).any()
    assert set(measurements.central_actionable_n) == {2}
    assert set(measurements.any_major_issue) == {False}
    assert set(measurements.any_major_contradiction) == {False}
    if first_coverage == "addressed":
        assert set(measurements.central_full_coverage_rate) == {1.0}
        assert set(measurements.central_missing_rate) == {0.0}
    else:
        assert measurements.central_full_coverage_rate.isna().all()
        assert measurements.central_missing_rate.isna().all()


def test_packet_measurements_and_concern_audit(monkeypatch):
    source = reference_input_from_packet(packet(), "126")
    template = evaluability_inventory(source)["reference_concerns"][0]
    definitions = [
        ("critical", "central", "uncertain", "evaluable", "missing"),
        ("developmental", "central", "supported", "partially_evaluable", "partially_addressed"),
        ("developmental", "central", "supported", "evaluable", "addressed"),
        ("critical", "central", "supported", "evaluable", "unable_to_judge"),
        ("positive", "central", "supported", "evaluable", "missing"),
        ("critical", "secondary", "supported", "evaluable", "missing"),
        ("critical", "central", "unsupported", "evaluable", "missing"),
        ("developmental", "central", "supported", "not_evaluable", "not_applicable"),
    ]
    concerns, coverage = [], []
    for index, (kind, centrality, support, evaluability, outcome) in enumerate(definitions, 1):
        identifier = f"R{index}"
        concerns.append({
            **template, "concern_id": identifier, "concern_type": kind,
            "centrality": centrality, "reference_support": support, "evaluability": evaluability,
        })
        coverage.append({
            "concern_id": identifier, "coverage": outcome, "rationale": "Coverage rationale.",
            "generated_references": [], "evidence_segment_ids": ["s1"],
        })
    protocol = {
        "feedback_prompt_version": "0.12.0", "judge_repetitions": 2,
        "reference_inventories": {"inventory": {"inventory": {"reference_concerns": concerns}}},
        "entries": [{
            "identity": {"document_id": "126", "condition": "G2"}, "generation_repetition": 3,
            "packet": {"packet_id": "packet", "reference_inventory_ref": {
                "inventory_id": "inventory", "inventory_sha256": "hash", "reflection_id": "126",
            }},
        }],
    }
    output = {
        "concern_coverage": coverage, "overall_severity": "major", "rationale": "Summary.",
        "reference_uncertainties": [],
        "issues": [
            {"issue_type": "central_omission", "severity": "major", "concern_ids": ["R1", "R2"]},
            {"issue_type": "contradiction", "severity": "minor", "concern_ids": []},
        ],
    }
    results = {("packet", 1): {"judgment": {"result": output}}}
    monkeypatch.setattr(stability, "study_results", lambda directory: ({"protocol": protocol}, results))
    tables = stability.extract_feedback_measurements("unused")
    rows = tables["measurements"]
    assert list(rows) == stability.FEEDBACK_MEASUREMENT_COLUMNS
    first = rows.iloc[0]
    assert (first.reflection_id, first.packet_id, first.condition) == ("126", "packet", "G2")
    assert (first.generation_repetition, first.judge_repetition) == (3, 1)
    assert first.central_actionable_n == 4
    assert first.central_missing_n == 1
    assert first.central_partial_n == 1
    assert first.central_missing_rate == pytest.approx(1 / 3)
    assert first.central_full_coverage_rate == pytest.approx(1 / 3)
    assert first.central_addressed_n == 1
    assert first.central_rate_denominator_n == 3
    assert first.any_major_issue
    assert not first.any_major_contradiction
    # Excluded concerns (unable_to_judge, secondary, unsupported, not_evaluable) stay out of micro denominators.
    micro = stability.summarize_micro_feedback_metrics(rows).iloc[0]
    assert micro.central_denominator_n == 3 and micro.positive_denominator_n == 1
    assert (micro.CMR, micro.CPCR, micro.CFCR) == (33.33, 33.33, 33.33)
    assert micro.PRR == 0.0
    assert (micro.MDI, micro.evaluated_feedbacks_n) == (100.0, 1)
    pending = rows.iloc[1]
    assert pending.central_actionable_n == 4
    assert pending.overall_severity == "missing"
    assert pending.positive_eligible_n == 1
    assert pd.isna(pending.positive_addressed_n)
    assert pd.isna(pending.positive_rate_denominator_n)
    assert pd.isna(pending.positive_recognition_rate)
    assert rows.loc[[1], ["central_missing_n", "central_partial_n", "central_missing_rate",
                          "central_full_coverage_rate", "any_major_issue",
                          "any_major_contradiction"]].isna().all().all()
    audit = tables["concern_coverage"]
    assert len(audit) == 16
    judged = audit[audit.judge_repetition == 1].set_index("concern_id")
    assert judged.central_actionable.sum() == 4
    assert judged.in_coverage_rate_denominator.sum() == 3
    assert judged.loc["R4", "coverage"] == "unable_to_judge"
    assert judged.loc["R8", "coverage"] == "not_applicable"
    assert judged.loc["R1", "linked_issue_indices"] == [0]
    assert judged.loc["R1", "coverage_evidence_segment_ids"] == ["s1"]
    assert judged.loc["R2", "evaluable_scope"] == template["evaluable_scope"]
    assert not audit[audit.judge_repetition == 2].in_coverage_rate_denominator.any()
    # Changing Issue grouping cannot change concern-based measures.
    output["issues"] = [
        {"issue_type": "central_omission", "severity": "major", "concern_ids": [c]}
        for c in ("R1", "R2")
    ]
    regrouped = stability.extract_feedback_measurements("unused")["measurements"]
    assert rows.equals(regrouped)
    output["issues"].append({"issue_type": "contradiction", "severity": "major", "concern_ids": []})
    assert stability.extract_feedback_measurements("unused")["measurements"].iloc[0].any_major_contradiction
    output["issues"] = []
    output["overall_severity"] = "no_error"
    assert not stability.extract_feedback_measurements("unused")["measurements"].iloc[0].any_major_issue
    for concern in concerns:
        concern["reference_support"] = "unsupported"
    empty = stability.extract_feedback_measurements("unused")["measurements"]
    assert set(empty.central_actionable_n) == {0}
    assert empty.iloc[0].central_missing_n == 0
    assert empty.central_missing_rate.isna().all()
    assert empty.central_full_coverage_rate.isna().all()


@pytest.mark.parametrize("centrality", ["central", "secondary"])
@pytest.mark.parametrize("support", ["supported", "uncertain", "unsupported"])
@pytest.mark.parametrize("evaluability", ["evaluable", "partially_evaluable", "not_evaluable"])
@pytest.mark.parametrize("outcome", ["addressed", "partially_addressed", "missing", "unable_to_judge"])
def test_positive_recognition_eligibility(monkeypatch, centrality, support, evaluability, outcome):
    concern = {
        "concern_id": "R1", "concern_type": "positive", "centrality": centrality,
        "reference_support": support, "evaluability": evaluability,
    }
    protocol = {
        "feedback_prompt_version": "0.12.0", "judge_repetitions": 1,
        "reference_inventories": {"i": {"inventory": {"reference_concerns": [concern]}}},
        "entries": [{
            "identity": {"document_id": "3", "condition": "G1"}, "generation_repetition": 1,
            "packet": {"packet_id": "p", "reference_inventory_ref": {
                "inventory_id": "i", "reflection_id": "3",
            }},
        }],
    }
    output = {
        "overall_severity": "no_error", "rationale": "r", "reference_uncertainties": [], "issues": [],
        "concern_coverage": [{
            "concern_id": "R1",
            "coverage": "not_applicable" if evaluability == "not_evaluable" else outcome,
            "rationale": "r", "generated_references": [], "evidence_segment_ids": [],
        }],
    }
    monkeypatch.setattr(stability, "study_results", lambda _: (
        {"protocol": protocol}, {("p", 1): {"judgment": {"result": output}}},
    ))
    tables = stability.extract_feedback_measurements("unused")
    row = tables["measurements"].iloc[0]
    eligible = support != "unsupported" and evaluability != "not_evaluable"
    denominator = eligible and outcome != "unable_to_judge"
    assert row.positive_eligible_n == int(eligible)
    assert row.positive_rate_denominator_n == int(denominator)
    assert row.positive_addressed_n == int(denominator and outcome == "addressed")
    if denominator:
        assert row.positive_recognition_rate == float(outcome == "addressed")
    else:
        assert pd.isna(row.positive_recognition_rate)
    assert row.central_actionable_n == 0
    audit = tables["concern_coverage"].iloc[0]
    assert audit.positive_eligible == eligible
    assert audit.in_positive_rate_denominator == denominator


def test_four_metric_macro_summary_missingness_and_repetitions():
    rows = pd.DataFrame({
        "condition": ["G1"] * 4, "reflection_id": ["1", "2", "3", "1"],
        "generation_repetition": [1] * 4, "judge_repetition": [1, 1, 1, 2],
        "positive_recognition_rate": [1.0, 0.5, None, 0.0],
        "central_missing_rate": [0.0, 1.0, None, 0.0],
        "central_full_coverage_rate": [1.0, 0.0, None, 1.0],
        "any_major_issue": pd.array([True, False, None, False], dtype="boolean"),
    })
    summary = stability.summarize_feedback_metrics(rows).set_index("judge_repetition")
    first = summary.loc[1]
    assert first.reflections_n == 3
    assert first.prr_reflections_n == 2
    assert first.positive_recognition_rate == 0.75
    assert first.central_missing_rate == 0.5
    assert first.central_full_coverage_rate == 0.5
    assert first.evaluated_feedbacks_n == 2
    assert first.major_discrepancy_feedbacks_n == 1
    assert first.major_discrepancy_incidence == 0.5
    assert summary.loc[2].major_discrepancy_incidence == 0
    assert summary.loc[2].positive_recognition_rate == 0
    with pytest.raises(ValueError, match="one packet per reflection"):
        stability.summarize_feedback_metrics(pd.concat([rows, rows.iloc[[0]]]))
    rows.loc[:, "positive_recognition_rate"] = float("nan")
    rows.loc[:, "any_major_issue"] = pd.NA
    undefined = stability.summarize_feedback_metrics(rows)
    assert undefined.positive_recognition_rate.isna().all()
    assert undefined.major_discrepancy_incidence.isna().all()
    assert set(undefined.evaluated_feedbacks_n) == {0}


def micro_rows(**overrides):
    data = {
        "condition": ["G1", "G1", "G2", "G3"], "reflection_id": ["1", "2", "1", "1"],
        "generation_repetition": [1] * 4, "judge_repetition": [1] * 4,
        "positive_addressed_n": [3, 0, 1, 0], "positive_rate_denominator_n": [4, 1, 3, 0],
        "central_missing_n": [1, 0, 2, 0], "central_partial_n": [1, 0, 1, 0],
        "central_addressed_n": [2, 1, 0, 0], "central_rate_denominator_n": [4, 1, 3, 0],
        "any_major_issue": [True, True, False, False],
    }
    data.update(overrides)
    frame = pd.DataFrame(data)
    for column in [c for c in data if c.endswith("_n")]:
        frame[column] = frame[column].astype("Int64")
    frame["any_major_issue"] = frame["any_major_issue"].astype("boolean")
    return frame


def test_micro_metrics_pool_counts_before_division():
    summary = stability.summarize_micro_feedback_metrics(micro_rows()).set_index("Generator")
    assert list(summary.reset_index())[:8] == [
        "Generator", "generation_repetition", "judge_repetition", "PRR", "CMR", "CPCR", "CFCR", "MDI",
    ]
    g1 = summary.loc["G1"]
    # Micro PRR = 3/5 = 60.00; the macro mean of 0.75 and 0.0 would be 37.5.
    assert (g1.PRR, g1.positive_addressed_n, g1.positive_denominator_n) == (60.0, 3, 5)
    assert (g1.CMR, g1.CPCR, g1.CFCR) == (20.0, 20.0, 60.0)
    assert (g1.central_missing_n, g1.central_partial_n, g1.central_addressed_n, g1.central_denominator_n) == (1, 1, 3, 5)
    assert (g1.MDI, g1.major_discrepancy_feedbacks_n, g1.evaluated_feedbacks_n) == (100.0, 2, 2)
    g2 = summary.loc["G2"]
    assert (g2.PRR, g2.CMR, g2.CPCR, g2.CFCR, g2.MDI) == (33.33, 66.67, 33.33, 0.0, 0.0)


def test_micro_coverage_distribution_identity():
    summary = stability.summarize_micro_feedback_metrics(micro_rows())
    defined = summary[summary.central_denominator_n > 0]
    assert ((defined.central_missing_n + defined.central_partial_n + defined.central_addressed_n)
            == defined.central_denominator_n).all()
    assert (defined[["CMR", "CPCR", "CFCR"]].sum(axis=1) - 100).abs().max() <= 0.02
    with pytest.raises(ValueError, match="partition"):
        stability.summarize_micro_feedback_metrics(micro_rows(central_addressed_n=[3, 1, 0, 0]))


def test_micro_metrics_zero_denominators_and_unexecuted_slots():
    summary = stability.summarize_micro_feedback_metrics(micro_rows()).set_index("Generator")
    g3 = summary.loc["G3"]
    assert pd.isna(g3.PRR) and pd.isna(g3.CMR) and pd.isna(g3.CPCR) and pd.isna(g3.CFCR)
    assert (g3.positive_denominator_n, g3.central_denominator_n, g3.MDI) == (0, 0, 0.0)
    pending = micro_rows(
        positive_addressed_n=[3, None, 1, 0], positive_rate_denominator_n=[4, None, 3, 0],
        central_missing_n=[1, None, 2, 0], central_partial_n=[1, None, 1, 0],
        central_addressed_n=[2, None, 0, 0], central_rate_denominator_n=[4, None, 3, 0],
        any_major_issue=[True, None, False, None],
    )
    summary = stability.summarize_micro_feedback_metrics(pending).set_index("Generator")
    assert (summary.loc["G1"].PRR, summary.loc["G1"].evaluated_feedbacks_n) == (75.0, 1)
    assert pd.isna(summary.loc["G3"].MDI) and summary.loc["G3"].evaluated_feedbacks_n == 0
    with pytest.raises(ValueError, match="one packet per reflection"):
        stability.summarize_micro_feedback_metrics(pd.concat([pending, pending.iloc[[0]]]))


def test_micro_mdi_uses_judge_major_flag_not_missing_concerns():
    rows = micro_rows(central_missing_n=[4, 1, 3, 0], central_partial_n=[0, 0, 0, 0],
                      central_addressed_n=[0, 0, 0, 0], any_major_issue=[False, False, False, False])
    summary = stability.summarize_micro_feedback_metrics(rows).set_index("Generator")
    assert summary.loc["G1"].CMR == 100.0
    assert summary.loc["G1"].MDI == 0.0


def evaluability_inventory(source):
    data = inventory_data(source)
    data["schema_version"] = "1.1.0"
    data["reference_concerns"][0].update(
        evaluability="partially_evaluable", evaluability_rationale="Current work is available.",
        evaluable_scope="Improve literature support in the current argument.",
        unavailable_context=["Earlier reflection version"],
    )
    extra = copy.deepcopy(data["reference_concerns"][0])
    extra.update(concern_id="R2", evaluability="not_evaluable", evaluable_scope=None)
    data["reference_concerns"].append(extra)
    return data


def test_evaluability_filter_scope_and_version_compatibility():
    source = reference_input_from_packet(packet(), "126")
    data = evaluability_inventory(source)
    p = with_inventory(packet(), data)
    payload = build_judge_input(p, feedback_prompt_version="0.9.0")
    assert len(payload["reference_inventory"]["reference_concerns"]) == 1
    assert payload["reference_inventory"]["reference_concerns"][0]["evaluable_scope"] == (
        "Improve literature support in the current argument."
    )
    result = frozen_output()
    result["schema_version"] = "2.1.0"
    validate_frozen_concern_output(EvaluableConcernJudgeOutput.model_validate(result), p)
    model = FakeChatModel(result)
    record = run_quality_judge(
        model=model, packet=p, provider="test", model_name="fake", feedback_prompt_version="0.9.0",
    )
    assert record["schema_version"] == "2.1.0"
    assert record["reference_inventory_ref"] == p["reference_inventory_ref"]
    assert data["reference_concerns"][1]["evaluability"] == "not_evaluable"
    result["concern_coverage"].append({
        **result["concern_coverage"][0], "concern_id": "R2",
    })
    with pytest.raises(ValueError, match="every supplied"):
        validate_frozen_concern_output(EvaluableConcernJudgeOutput.model_validate(result), p)
    with pytest.raises(ValueError, match="schema 1.1.0"):
        build_judge_input(with_inventory(packet(), inventory_data(source)), feedback_prompt_version="0.9.0")
    with pytest.raises(ValueError, match="requires feedback prompt 0.9.0"):
        build_judge_input(p, feedback_prompt_version="0.8.0")


@pytest.mark.parametrize("evaluability,scope,context", [
    ("evaluable", None, []),
    ("partially_evaluable", "Current argument", []),
    ("not_evaluable", "Current argument", ["Earlier version"]),
    ("not_evaluable", None, []),
])
def test_evaluability_requires_consistent_scope_and_context(evaluability, scope, context):
    data = evaluability_inventory(reference_input_from_packet(packet(), "126"))
    data["reference_concerns"][0].update(
        evaluability=evaluability, evaluable_scope=scope, unavailable_context=context,
    )
    with pytest.raises(ValidationError):
        EvaluableConcernInventory.model_validate(data)


def test_evaluability_independent_of_reference_support():
    data = evaluability_inventory(reference_input_from_packet(packet(), "126"))
    data["reference_concerns"][0].update(
        evaluability="evaluable", reference_support="uncertain", unavailable_context=[],
    )
    parsed = EvaluableConcernInventory.model_validate(data)
    assert parsed.reference_concerns[0].evaluability == "evaluable"
    assert parsed.reference_concerns[0].reference_support == "uncertain"


def test_excluded_concerns_cannot_become_issues_and_empty_eligible_set_is_valid():
    source = reference_input_from_packet(packet(), "126")
    data = evaluability_inventory(source)
    p = with_inventory(packet(), data)
    result = frozen_output()
    result["schema_version"] = "2.1.0"
    result["issues"] = [{
        "issue_type": "central_omission", "severity": "major", "concern_ids": ["R2"],
        "human_excerpt": source["reference_teacher_feedback"], "generated_references": [],
        "evidence_segment_ids": [], "rationale": "Excluded comparison is absent.",
        "consequence": "Should never count against generated feedback.",
    }]
    result["overall_severity"] = "major"
    with pytest.raises(ValueError, match="unknown inventory"):
        validate_frozen_concern_output(EvaluableConcernJudgeOutput.model_validate(result), p)


def applicable_output():
    data = frozen_output()
    data["schema_version"] = "2.2.0"
    data["concern_coverage"].append({
        "concern_id": "R2", "coverage": "not_applicable",
        "generated_references": [], "evidence_segment_ids": [],
        "rationale": "Earlier version unavailable to the generator.",
    })
    return data


def test_not_applicable_preserves_full_inventory_and_no_error():
    p = with_inventory(packet(), evaluability_inventory(reference_input_from_packet(packet(), "126")))
    data = applicable_output()
    validate_frozen_concern_output(ApplicableConcernJudgeOutput.model_validate(data), p)
    payload = build_judge_input(p, feedback_prompt_version="0.10.0")
    assert payload["reference_inventory"] == p["reference_inventory"]
    record = run_quality_judge(
        model=FakeChatModel(data), packet=p, provider="test", model_name="fake",
        feedback_prompt_version="0.10.0",
    )
    assert record["schema_version"] == "2.2.0"
    assert record["reference_inventory_ref"] == p["reference_inventory_ref"]
    assert record["result"]["overall_severity"] == "no_error"


def priority_output(source, concern_id="R1"):
    data = applicable_output()
    data.update(schema_version="2.3.0", overall_severity="major")
    data["issues"] = [{
        "issue_type": "central_omission", "severity": "major", "concern_ids": [concern_id],
        "severity_basis": "documented_reference_deviation",
        "human_excerpt": source["reference_teacher_feedback"], "generated_references": [],
        "evidence_segment_ids": [],
        "rationale": "A central documented lecturer priority is missing; its disciplinary validity remains uncertain.",
        "consequence": "Major deviation from the lecturer priority, not independently confirmed disciplinary error.",
    }]
    return data


def test_refined_prompt_retains_shape_and_requires_non_circular_reference_relative_severity():
    text = resolve_prompt("feedback_quality_llm_judge", "0.12.0").text
    assert "nicht automatisch einen unabhängig bestätigten pädagogischen Qualitätsmangel" in text
    assert "die Zentralität oder Nichtabdeckung allein genügt nicht" in text
    assert "anhand des konkreten Feedbacks" in text
    assert "Abdeckung einzelner Referenzanliegen wird unabhängig von der Gruppierung" in text
    assert "central_omission zusätzlich ausschließlich zentrale Anliegen" in text
    assert (
        ReferencePriorityJudgeOutput.model_json_schema()["properties"]
        == RefinedReferencePriorityJudgeOutput.model_json_schema()["properties"]
    )


@pytest.mark.parametrize("evaluability", ["evaluable", "partially_evaluable"])
@pytest.mark.parametrize("severity", ["major", "minor"])
@pytest.mark.parametrize("version", ["0.11.0", "0.12.0"])
def test_uncertain_central_reference_can_justify_issue_without_mutation(evaluability, severity, version):
    source = reference_input_from_packet(packet(), "126")
    inventory = evaluability_inventory(source)
    inventory["reference_concerns"][0].update(
        reference_support="uncertain", evaluability=evaluability,
    )
    p = with_inventory(packet(), inventory)
    before = copy.deepcopy(p)
    data = priority_output(source)
    data["issues"][0]["severity"] = severity
    data["overall_severity"] = severity
    record = run_quality_judge(
        model=FakeChatModel(data), packet=p, provider="test", model_name="fake",
        feedback_prompt_version=version,
    )
    assert record["schema_version"] == "2.3.0"
    assert record["result"]["issues"][0]["severity_basis"] == "documented_reference_deviation"
    assert record["reference_inventory_ref"] == p["reference_inventory_ref"]
    assert p == before
    # The same omission remains invalid under the historical supported-only policy.
    old = copy.deepcopy(data)
    old["schema_version"] = "2.2.0"
    del old["issues"][0]["severity_basis"]
    with pytest.raises(ValueError, match="supported inventory"):
        validate_frozen_concern_output(ApplicableConcernJudgeOutput.model_validate(old), p)


@pytest.mark.parametrize("change", ["unsupported", "secondary", "not_evaluable", "wrong_basis", "no_basis"])
def test_priority_issue_eligibility_is_enforced(change):
    source = reference_input_from_packet(packet(), "126")
    inventory = evaluability_inventory(source)
    inventory["reference_concerns"][0]["reference_support"] = "uncertain"
    data = priority_output(source)
    if change == "unsupported":
        inventory["reference_concerns"][0]["reference_support"] = "unsupported"
    elif change == "secondary":
        inventory["reference_concerns"][0]["centrality"] = "secondary"
    elif change == "not_evaluable":
        data["issues"][0]["concern_ids"] = ["R2"]
    elif change == "wrong_basis":
        data["issues"][0]["severity_basis"] = "independent_generated_error"
    else:
        del data["issues"][0]["severity_basis"]
    with pytest.raises(ValueError):
        validate_frozen_concern_output(
            ReferencePriorityJudgeOutput.model_validate(data), with_inventory(packet(), inventory),
        )


@pytest.mark.parametrize("support", ["supported", "uncertain"])
@pytest.mark.parametrize("evaluability", ["evaluable", "partially_evaluable"])
@pytest.mark.parametrize("issue_type", ["secondary_omission", "contradiction", "misleading_emphasis"])
def test_refined_secondary_concerns_can_justify_minor_but_not_major(support, evaluability, issue_type):
    source = reference_input_from_packet(packet(), "126")
    inventory = evaluability_inventory(source)
    inventory["reference_concerns"][0].update(
        centrality="secondary", reference_support=support, evaluability=evaluability,
    )
    p = with_inventory(packet(), inventory)
    before = copy.deepcopy(p)
    data = priority_output(source)
    data["overall_severity"] = "minor"
    issue = data["issues"][0]
    issue.update(
        issue_type=issue_type, severity="minor",
        rationale="The advice omits a concrete secondary lecturer qualification.",
        consequence="The proposed next step loses a bounded qualification, not the main direction.",
    )
    if issue_type != "secondary_omission":
        issue["generated_references"] = [{
            "component": "strengths", "item_index": 0, "excerpt": "Your description is clear.",
        }]
    record = run_quality_judge(
        model=FakeChatModel(data), packet=p, provider="test", model_name="fake",
        feedback_prompt_version="0.12.0",
    )
    assert record["schema_version"] == "2.3.0"
    assert record["prompt_version"] == "0.12.0"
    assert p == before
    if support == "uncertain":
        with pytest.raises(ValueError, match="require central concerns"):
            validate_frozen_concern_output(ReferencePriorityJudgeOutput.model_validate(data), p)
    issue["severity"] = "major"
    data["overall_severity"] = "major"
    with pytest.raises(ValueError):
        validate_frozen_concern_output(RefinedReferencePriorityJudgeOutput.model_validate(data), p)


@pytest.mark.parametrize("change", ["unsupported", "not_evaluable", "central_omission"])
def test_refined_secondary_policy_preserves_exclusions(change):
    source = reference_input_from_packet(packet(), "126")
    inventory = evaluability_inventory(source)
    inventory["reference_concerns"][0].update(centrality="secondary", reference_support="uncertain")
    data = priority_output(source)
    data["overall_severity"] = "minor"
    data["issues"][0].update(issue_type="secondary_omission", severity="minor")
    if change == "unsupported":
        inventory["reference_concerns"][0]["reference_support"] = "unsupported"
    elif change == "not_evaluable":
        data["issues"][0]["concern_ids"] = ["R2"]
    else:
        data["issues"][0]["issue_type"] = "central_omission"
    with pytest.raises(ValueError):
        validate_frozen_concern_output(
            RefinedReferencePriorityJudgeOutput.model_validate(data), with_inventory(packet(), inventory),
        )


@pytest.mark.parametrize("support", ["uncertain", "unsupported"])
@pytest.mark.parametrize("centrality", ["central", "secondary"])
@pytest.mark.parametrize("output_model", [ReferencePriorityJudgeOutput, RefinedReferencePriorityJudgeOutput])
def test_uncertain_or_unsupported_missing_coverage_is_not_automatically_an_issue(support, centrality, output_model):
    source = reference_input_from_packet(packet(), "126")
    inventory = evaluability_inventory(source)
    inventory["reference_concerns"][0].update(reference_support=support, centrality=centrality)
    p = with_inventory(packet(), inventory)
    data = applicable_output()
    data["schema_version"] = "2.3.0"
    validate_frozen_concern_output(output_model.model_validate(data), p)
    assert data["overall_severity"] == "no_error"


def test_uncertain_reference_contradiction_and_independent_error_have_distinct_bases():
    source = reference_input_from_packet(packet(), "126")
    inventory = evaluability_inventory(source)
    inventory["reference_concerns"][0]["reference_support"] = "uncertain"
    p = with_inventory(packet(), inventory)
    data = priority_output(source)
    issue = data["issues"][0]
    issue.update(
        issue_type="contradiction",
        generated_references=[{
            "component": "strengths", "item_index": 0, "excerpt": "Your description is clear.",
        }],
        evidence_segment_ids=["s1"],
    )
    validate_frozen_concern_output(ReferencePriorityJudgeOutput.model_validate(data), p)
    issue.update(
        concern_ids=[], human_excerpt=None, severity_basis="independent_generated_error",
    )
    validate_frozen_concern_output(ReferencePriorityJudgeOutput.model_validate(data), p)
    issue["severity_basis"] = "documented_reference_deviation"
    with pytest.raises(ValueError, match="Independent generated errors"):
        validate_frozen_concern_output(ReferencePriorityJudgeOutput.model_validate(data), p)


@pytest.mark.parametrize("change", ["missing", "unknown", "omit", "eligible_na", "na_evidence", "issue"])
def test_not_applicable_enforced_against_frozen_evaluability(change):
    p = with_inventory(packet(), evaluability_inventory(reference_input_from_packet(packet(), "126")))
    data = applicable_output()
    if change == "missing":
        data["concern_coverage"][1]["coverage"] = "missing"
    elif change == "unknown":
        data["concern_coverage"][1]["concern_id"] = "R3"
    elif change == "omit":
        data["concern_coverage"].pop()
    elif change == "eligible_na":
        data["concern_coverage"][0]["coverage"] = "not_applicable"
    elif change == "na_evidence":
        data["concern_coverage"][1]["evidence_segment_ids"] = ["s1"]
    else:
        data["issues"] = [{
            "issue_type": "unsupported_claim", "severity": "major", "concern_ids": ["R2"],
            "human_excerpt": None, "generated_references": [{
                "component": "strengths", "item_index": 0, "excerpt": "Your description is clear.",
            }],
            "evidence_segment_ids": ["s1"], "rationale": "Excluded reference comparison.",
            "consequence": "Must not contribute severity.",
        }]
        data["overall_severity"] = "major"
    with pytest.raises(ValueError):
        validate_frozen_concern_output(ApplicableConcernJudgeOutput.model_validate(data), p)


@pytest.mark.parametrize("issue_type", ["unsupported_claim", "contradiction"])
def test_independent_generated_errors_remain_judgeable(issue_type):
    p = with_inventory(packet(), evaluability_inventory(reference_input_from_packet(packet(), "126")))
    for c in p["reference_inventory"]["reference_concerns"]:
        c.update(evaluability="not_evaluable", evaluable_scope=None)
    p["reference_inventory_ref"]["inventory_sha256"] = content_hash(p["reference_inventory"])
    data = applicable_output()
    data["concern_coverage"][0]["coverage"] = "not_applicable"
    data["issues"] = [{
        "issue_type": issue_type, "severity": "major", "concern_ids": [],
        "human_excerpt": None, "generated_references": [{
            "component": "strengths", "item_index": 0, "excerpt": "Your description is clear.",
        }],
        "evidence_segment_ids": ["s1"], "rationale": "An independently demonstrated generated error.",
        "consequence": "The generated assessment direction is materially wrong.",
    }]
    data["overall_severity"] = "major"
    validate_frozen_concern_output(ApplicableConcernJudgeOutput.model_validate(data), p)
    assert build_judge_input(p, feedback_prompt_version="0.10.0")["reference_inventory"] == p["reference_inventory"]


def test_legacy_evaluability_empty_eligible_set():
    data = evaluability_inventory(reference_input_from_packet(packet(), "126"))
    data["reference_concerns"][0].update(
        evaluability="not_evaluable", evaluable_scope=None,
    )
    p = with_inventory(packet(), data)
    assert build_judge_input(p, feedback_prompt_version="0.9.0")["reference_inventory"]["reference_concerns"] == []
    result = frozen_output([])
    result["schema_version"] = "2.1.0"
    validate_frozen_concern_output(EvaluableConcernJudgeOutput.model_validate(result), p)


@pytest.mark.parametrize("document,phrases", [
    ("3", ["Analytische Tiefe", "Verschiedene Perspektiven", "Theoretische Fundierung", "Handlungsalternativen"]),
    ("126", ["Bewertungskompetenz", "Weitere Literatur"]),
])
def test_playground_concern_boundaries_are_representable(document, phrases):
    # Small synthetic boundary fixtures; private full documents are exercised by the pilot.
    p = packet()
    p["reference_teacher_feedback"] = ". ".join(phrases)
    source = reference_input_from_packet(p, document)
    data = inventory_data(source)
    template = data["reference_concerns"][0]
    data["reference_concerns"] = [
        {**template, "concern_id": f"R{i}", "human_excerpt": phrase, "normalized_concern": phrase}
        for i, phrase in enumerate(phrases, 1)
    ]
    validate_inventory(ReferenceConcernInventory.model_validate(data), source)
    assert len({c["normalized_concern"] for c in data["reference_concerns"]}) == len(phrases)
