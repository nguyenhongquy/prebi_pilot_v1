import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from reflection_assessment_feedback import EXPERT_RUBRIC, make_generation_request
from reflection_assessment_feedback.generator import GenerationRun, RubricGenerationRunner
from reflection_assessment_feedback.judge_study import content_hash
from reflection_assessment_feedback.models import DocumentGenerationOutput, HumanAnalysisContext
from reflection_assessment_feedback.prompt import build_prompt, prompt_components_for_request
from scripts import regenerate_g3_playground as diagnostic
from test_document_pipeline import document, human_context, output_record
from test_llm_judge import FakeChatModel, _packet
from test_alignment_judge import output as alignment_output


def request():
    doc = document()
    doc.document_id = "3"
    return make_generation_request(
        condition="G3", document=doc, rubric=EXPERT_RUBRIC, analysis=human_context(),
    )


def test_opt_in_fragment_preserves_payload_and_defaults():
    req = request()
    before = req.model_dump(mode="json")
    old = build_prompt(req)
    new = build_prompt(req, g3_supplemental_version="2.1.0")
    assert old != new
    assert old.split("Arbeitsgrundlage (JSON):\n")[1] == new.split("Arbeitsgrundlage (JSON):\n")[1]
    assert req.model_dump(mode="json") == before
    assert prompt_components_for_request(req)[-1].version == "2.0.0"
    assert prompt_components_for_request(req, g3_supplemental_version="2.1.0")[-1].version == "2.1.0"
    assert "not_assigned schliessen die Textstelle nicht" in new
    assert "Konsistenz mit deiner evidence_synthesis" in new
    assert "document 3" not in new
    g1 = make_generation_request(condition="G1", document=req.document, rubric=EXPERT_RUBRIC)
    with pytest.raises(ValueError, match="G3 request"):
        build_prompt(g1, g3_supplemental_version="2.1.0")


@pytest.mark.parametrize("version", ["2.1.0", "2.2.0"])
def test_runner_records_revised_component_and_actual_prompt(version):
    req = request()

    class GenerationModel:
        prompt = ""

        def with_structured_output(self, schema: type[BaseModel]):
            assert schema is DocumentGenerationOutput
            return self

        def invoke(self, input: str, config: RunnableConfig | None = None):
            self.prompt = input
            return output_record()

    model = GenerationModel()
    runner = RubricGenerationRunner(model, g3_supplemental_version=version)
    result = runner._generate_traced(req, repetition=1)
    assert result.prompt_version == "2.0.0"
    assert result.prompt_components[-1]["version"] == version
    assert result.rendered_prompt_sha256 == hashlib.sha256(
        build_prompt(req, g3_supplemental_version=version).encode(),
    ).hexdigest()
    assert model.prompt == build_prompt(req, g3_supplemental_version=version)


def test_dimension_only_payload_whitelists_fields_and_unions_assignments():
    req = request()
    assert isinstance(req.analysis, HumanAnalysisContext)
    candidate = req.analysis.segments[0].candidates[1]
    candidate.component_bands["HA"] = "1"
    candidate.known_component_bands["HA"] = True
    before = req.model_dump(mode="json")
    rendered = build_prompt(req, g3_supplemental_version="2.2.0")
    payload = json.loads(rendered.split("Arbeitsgrundlage (JSON):\n")[1])
    assert set(payload) == {"rubric", "reflection", "segment_dimension_assignments"}
    assert payload["segment_dimension_assignments"] == [
        {"segment_id": "s1", "dimension_ids": ["SW", "HA"]},
        {"segment_id": "s2", "dimension_ids": ["SW"]},
    ]
    original = json.loads(build_prompt(req).split("Arbeitsgrundlage (JSON):\n")[1])
    assert payload["reflection"] == original["reflection"]
    assert payload["rubric"] == original["rubric"]
    for forbidden in (
        "candidate_id", "raw_scope_value", "raw_component_value", "requires_review",
        "conflicting_annotations", "known_component_bands", '"label"', '"source"',
        "band_counts", "annotation_count", "evidence_profile",
    ):
        assert forbidden not in rendered
    assert req.model_dump(mode="json") == before
    assert prompt_components_for_request(req)[-1].version == "2.0.0"


def test_dimension_only_ignores_levels_unknowns_and_scope():
    req = request()
    assert isinstance(req.analysis, HumanAnalysisContext)
    before = build_prompt(req, g3_supplemental_version="2.2.0")
    for segment in req.analysis.segments:
        for candidate in segment.candidates:
            candidate.candidate_id = "private-source-marker"
            candidate.requires_review = not candidate.requires_review
            candidate.component_bands["SW"] = "2"
            candidate.component_bands["UA"] = "3"
    assert build_prompt(req, g3_supplemental_version="2.2.0") == before
    for segment in req.analysis.segments:
        for candidate in segment.candidates:
            candidate.scope = "out_of_scope"
    payload = json.loads(build_prompt(
        req, g3_supplemental_version="2.2.0",
    ).split("Arbeitsgrundlage (JSON):\n")[1])
    assert all(not item["dimension_ids"] for item in payload["segment_dimension_assignments"])


@pytest.fixture
def source_study(tmp_path):
    req = request()
    assert req.analysis is not None
    config = diagnostic.load_experiment_config()
    artifact = {
        "g3_human_analysis_context": req.analysis.model_dump(mode="json"),
        "pairs": [{"repetition": 1, "G3": {
            "model_name": config["generation"]["model_name"], "output": output_record(),
            "rendered_prompt_sha256": hashlib.sha256(build_prompt(req).encode()).hexdigest(),
        }}],
    }
    text = json.dumps(artifact)
    digest = hashlib.sha256(text.encode()).hexdigest()
    packet = _packet("assessment_quality")
    packet.update(reflection={"segments": [s.model_dump(mode="json") for s in req.document.segments]},
                  rubric=req.rubric.model_dump(mode="json"))
    protocol = {
        "schema_version": "1.0.0", "cohort": "playground",
        "generation_sources": {digest: text},
        "entries": [{
            "identity": {"document_id": "3", "condition": "G3", "source_artifact_sha256": digest},
            "generation_repetition": 1, "packet": packet,
            "input_sha256": diagnostic.judge_input_sha256(packet),
        }],
    }
    identifier = content_hash(protocol)
    source = tmp_path / "source" / identifier
    source.mkdir(parents=True)
    (source / "manifest.json").write_text(json.dumps({"study_id": identifier, "protocol": protocol}))
    return tmp_path, source


@pytest.mark.parametrize("version", ["2.1.0", "2.2.0"])
def test_frozen_three_generation_run_validates_resumes_and_preserves_sources(source_study, monkeypatch, version):
    root, source = source_study
    original = (source / "manifest.json").read_bytes()
    directory, protocol = diagnostic.prepare(root, source, g3_supplemental_version=version)
    assert diagnostic.prepare(root, source, g3_supplemental_version=version)[0] == directory
    assert protocol["generation_repetitions"] == 3
    req = request()
    assert req.analysis is not None
    assert protocol["request"]["analysis"] == req.analysis.model_dump(mode="json")
    with pytest.raises(RuntimeError, match="approval"):
        diagnostic.run(directory, protocol)
    seen = []

    class Runner:
        def __init__(self, *args, **kwargs):
            assert kwargs["g3_supplemental_version"] == version

        def generate(self, req, repetition):
            seen.append(repetition)
            assert not list(directory.glob("alignment-*.json"))
            return GenerationRun(
                run_id=f"synthetic-{repetition}", condition="G3", repetition=repetition,
                model_name=protocol["original_model"], generation_parameters={},
                prompt_version="2.0.0", prompt_artifact_sha256=protocol["prompt_components"][0]["sha256"],
                rendered_prompt_sha256=protocol["rendered_prompt_sha256"],
                prompt_components=protocol["prompt_components"], rubric_artifact_sha256="synthetic",
                output=DocumentGenerationOutput.model_validate(output_record()), feedback_letter="Synthetic.",
            )

    actual_judge = diagnostic.run_quality_judge
    judge_calls = []

    def judge(**kwargs):
        assert seen == [1, 2, 3]
        judge_calls.append(kwargs["packet"]["packet_id"])
        data = alignment_output(1.5)
        return actual_judge(**{
            **kwargs, "model": FakeChatModel(data), "enable_langsmith_tracing": False,
        })

    monkeypatch.setattr(diagnostic, "RubricGenerationRunner", Runner)
    monkeypatch.setattr(diagnostic, "run_quality_judge", judge)
    monkeypatch.setattr(diagnostic, "reserve_request_slot", lambda *args: None)
    args = {
        "approve_external_processing": True, "approve_full_batch": True,
        "generation_model": SimpleNamespace(model=protocol["original_model"]),
        "judge_model": SimpleNamespace(model_name=protocol["judge_config"]["model_name"]),
    }
    for _ in range(2):
        rows = diagnostic.run(directory, protocol, **args)
        assert len(rows) == 9
    assert seen == [1, 2, 3] and len(judge_calls) == len(set(judge_calls)) == 3
    assert (source / "manifest.json").read_bytes() == original
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in directory.glob("*.json"))
    result = directory / "generation-2.json"
    data = json.loads(result.read_text())
    data["rendered_prompt_sha256"] = "changed"
    result.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="provenance mismatch"):
        diagnostic.run(directory, protocol, **args)


def test_reject_protocol_drift(source_study):
    root, source = source_study
    directory, protocol = diagnostic.prepare(root, source)
    changed = copy.deepcopy(protocol)
    changed["generation_repetitions"] = 4
    with pytest.raises(ValueError, match="protocol changed"):
        diagnostic.run(directory, changed, approve_external_processing=True, approve_full_batch=True)
