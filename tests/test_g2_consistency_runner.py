from dataclasses import dataclass
import hashlib
import json
from unittest.mock import Mock

import pandas as pd
import pytest

from reflection_assessment_feedback import EXPERT_RUBRIC, IntermediateAnalysis, make_generation_request
from reflection_assessment_feedback.models import GenerationOutput, ReflectionDocument, ReflectionSegment, SegmentAnalysis
from scripts import rerun_g2_generation_consistency as runner


def synthetic_context(tmp_path):
    document = ReflectionDocument(document_id="first", segments=[
        ReflectionSegment(segment_id="segment-one", text="Synthetic reflection passage", order=0),
    ])
    analysis = IntermediateAnalysis(source="predicted", segments=[
        SegmentAnalysis(segment_id="segment-one", scope="in_scope", component_bands={"SW": "1", "UA": "1", "HA": "1"}),
    ])
    request = make_generation_request(condition="G2", document=document, rubric=EXPERT_RUBRIC, analysis=analysis)
    output = GenerationOutput.model_validate({
        "assessment": {"component_assessments": [
            {"dimension_id": dimension, "band": 1.0, "justification": "Synthetic evidence", "evidence_segment_ids": ["segment-one"]}
            for dimension in ("SW", "UA", "HA")
        ]},
        "feedback": {"strengths": [{"message": "Synthetic strength", "evidence_segment_ids": ["segment-one"]}]},
    })
    context = {
        "config": {"generation": {"model_name": "fake-model", "temperature": 0,
                                  "enable_langsmith_tracing": False, "minimum_request_interval_seconds": 1}},
        "protected": tmp_path, "directory": tmp_path / "experiment", "protocol_hash": "protocol-one",
        "protocol": {"request_hashes": {"first": runner.canonical_hash(request.model_dump(mode="json"))},
                     "prompt_hashes": {"first": hashlib.sha256(runner.build_prompt(request).encode()).hexdigest()}},
        "units": pd.DataFrame([{"document_id": "first", "author_id": "one", "mode": "test"}]),
        "requests": {"first": request},
    }
    return context, output


def test_new_g2_repeat_runner_resumes_without_provider_calls(tmp_path, monkeypatch):
    import langchain_google_genai

    context, output = synthetic_context(tmp_path)

    @dataclass
    class FakeRun:
        output: GenerationOutput

    calls = Mock(side_effect=lambda *args: FakeRun(output))
    monkeypatch.setattr(runner, "prepare", lambda: context)
    monkeypatch.setattr(runner, "generate_validated", calls)
    monkeypatch.setattr(runner, "analyze", Mock())
    monkeypatch.setattr(runner, "RubricGenerationRunner", lambda *args, **kwargs: object())
    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", lambda **kwargs: object())
    monkeypatch.setattr(runner.sys, "argv", ["runner", "--phase", "generate", "--approve-external-processing"])

    runner.main()
    assert calls.call_count == 5
    paths = list((context["directory"] / "results").glob("*.json"))
    assert len(paths) == 5
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in paths)
    runner.main()
    assert calls.call_count == 5

    record = json.loads(paths[0].read_text())
    record["protocol_sha256"] = "wrong-protocol"
    with pytest.raises(ValueError, match="provenance mismatch"):
        runner.validate_new_record(record, context, "first", record["repetition"])


def test_new_g2_repeat_runner_rejects_invalid_evidence(tmp_path):
    context, output = synthetic_context(tmp_path)
    invalid = output.model_dump(mode="json")
    invalid["assessment"]["component_assessments"][0]["evidence_segment_ids"] = ["unknown"]
    record = {
        "protocol_sha256": context["protocol_hash"], "document_id": "first", "condition": "G2", "repetition": 1,
        "request_sha256": context["protocol"]["request_hashes"]["first"],
        "prompt_sha256": context["protocol"]["prompt_hashes"]["first"], "output": invalid,
    }
    with pytest.raises(ValueError, match="segment ID"):
        runner.validate_new_record(record, context, "first", 1)