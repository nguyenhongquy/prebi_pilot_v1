import pandas as pd
import pytest

from reflection_assessment_feedback.generation_consistency import consistency_tables, encode_score, paired_consistency_differences


def fixture_scores():
    units = pd.DataFrame([
        {"document_id": "a", "author_id": "author1", "mode": "playground"},
        {"document_id": "b", "author_id": "author2", "mode": "playground"},
    ])
    scores = pd.DataFrame([
        {"document_id": document, "condition": condition, "dimension_id": dimension,
         "repetition": repetition, "score": value}
        for document, value in (("a", 1.0), ("b", 2.0))
        for condition in ("G1", "G2", "G3") for dimension in ("SW", "UA", "HA")
        for repetition in range(1, 4)
    ])
    return units, scores


def test_perfect_agreement_and_alpha():
    units, scores = fixture_scores()
    result = consistency_tables(scores, units=units, repetitions=3)
    assert result["within"].pairwise_exact.eq(1).all()
    assert result["within"].alpha_ordinal.eq(1).all()
    assert result["within"].alpha_nominal.eq(1).all()
    assert result["between"].all_pairs_exact.eq(1).all()
    differences = paired_consistency_differences(result["per_document"], samples=20)
    assert differences.difference.eq(0).all()
    assert differences.lower_95.eq(0).all()


def test_pairs_missing_and_undefined_alpha():
    units, scores = fixture_scores()
    mask = scores.document_id.eq("a") & scores.condition.eq("G1") & scores.dimension_id.eq("SW")
    scores.loc[mask & scores.repetition.eq(3), "score"] = 2.0
    result = consistency_tables(scores, units=units, repetitions=3)
    detail = result["per_document"].query("document_id == 'a' and condition == 'G1' and dimension_id == 'SW'").iloc[0]
    assert detail.pairwise_exact == pytest.approx(1 / 3)
    assert not detail.unanimous
    missing = consistency_tables(scores.loc[~(mask & scores.repetition.eq(3))], units=units, repetitions=3)
    assert missing["within"].query("condition == 'G1' and dimension_id == 'SW'").missing_ratings.item() == 1
    constant = consistency_tables(scores.assign(score=1.0), units=units, repetitions=3)
    assert constant["within"].alpha_ordinal.isna().all()
    assert constant["within"].alpha_ordinal_status.eq("no_category_variation").all()


def test_no_arbitrary_rounding_or_duplicate_ratings():
    assert encode_score(1.7) == "1.7"
    with pytest.raises(ValueError, match="explicit mapping"):
        encode_score(1.7, {"1.0": "1"})
    units, scores = fixture_scores()
    with pytest.raises(ValueError, match="Duplicate"):
        consistency_tables(pd.concat([scores, scores.iloc[:1]]), units=units)


def test_notebook_runner_resumes_without_new_calls(tmp_path, monkeypatch):
    import hashlib
    import json
    import os
    from dataclasses import dataclass
    from pathlib import Path

    import langchain_google_genai
    from reflection_assessment_feedback import EXPERT_RUBRIC
    from reflection_assessment_feedback.models import (
        GenerationRequest, GenerationOutput, ReflectionDocument, ReflectionSegment,
    )
    from reflection_assessment_feedback.generator import validate_output
    from reflection_assessment_feedback.prompt import build_prompt
    from reflection_assessment_feedback import rate_limit

    notebook = json.loads((Path(__file__).resolve().parents[1] / '10_generation_consistency.ipynb').read_text())
    source = next(''.join(cell['source']) for cell in notebook['cells']
                  if cell['cell_type'] == 'code' and 'def validate_saved_record' in ''.join(cell['source']))
    document = ReflectionDocument(document_id='a', segments=[
        ReflectionSegment(segment_id='s', text='Reflection passage', order=0),
    ])
    request = GenerationRequest(condition='G1', document=document, rubric=EXPERT_RUBRIC)
    output = GenerationOutput.model_validate({
        'assessment': {'component_assessments': [
            {'dimension_id': dimension, 'band': 1.0, 'justification': 'Evidence', 'evidence_segment_ids': ['s']}
            for dimension in ('SW', 'UA', 'HA')
        ]},
        'feedback': {'strengths': [{'message': 'Strength', 'evidence_segment_ids': ['s']}]},
    })
    snapshots = {('a', condition): request.model_dump(mode='json') for condition in ('G1', 'G2', 'G3')}

    def canonical_hash(value):
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    protocol = {
        'request_hashes': {f'a:{condition}': canonical_hash(snapshot) for (_, condition), snapshot in snapshots.items()},
        'prompt_hashes': {f'a:{condition}': hashlib.sha256(build_prompt(request).encode()).hexdigest()
                          for condition in ('G1', 'G2', 'G3')},
    }
    calls = []

    @dataclass
    class FakeRun:
        output: GenerationOutput

    class FakeRunner:
        def __init__(self, model, *, enable_langsmith_tracing):
            pass

        def generate(self, request, *, repetition):
            calls.append(repetition)
            if len(calls) == 1:
                raise ValueError('Invalid fake response')
            return FakeRun(output=output)

    monkeypatch.setattr(langchain_google_genai, 'ChatGoogleGenerativeAI', lambda **options: object())
    monkeypatch.setattr(rate_limit, 'reserve_request_slot', lambda *args: 0)
    monkeypatch.setenv('GOOGLE_API_KEY', 'fake-test-key')
    namespace = {
        'RUN_GENERATION': True, 'APPROVE_EXTERNAL_PROCESSING': True,
        'APPROVE_CATEGORY_POLICY': True, 'APPROVE_G2_PROVENANCE': True,
        'MODES_TO_RUN': ('playground',), 'PROTOCOL_FROZEN_FOR_TEST': False,
        'GENERATION_CONFIG': {'provider': 'google', 'model_name': 'fake', 'temperature': 0,
                              'minimum_request_interval_seconds': 1},
        'ENABLE_LANGSMITH_TRACING': False, 'canonical_hash': canonical_hash,
        'protocol': protocol, 'protocol_hash': canonical_hash(protocol),
        'EXPERIMENT_DIRECTORY': tmp_path / 'experiment', 'PROTECTED_ROOT': tmp_path,
        'request_snapshots': snapshots, 'Path': Path, 'hashlib': hashlib, 'os': os, 'json': json,
        'GenerationRequest': GenerationRequest, 'GenerationOutput': GenerationOutput,
        'validate_output': validate_output, 'build_prompt': build_prompt,
        'RubricGenerationRunner': FakeRunner,
        'cohort_units': pd.DataFrame([{'document_id': 'a', 'mode': 'playground'}]),
        'REPETITIONS': 2, 'MAX_ATTEMPTS': 3,
    }
    from datetime import datetime, timezone
    from uuid import uuid4
    namespace.update(datetime=datetime, timezone=timezone, uuid4=uuid4)
    exec(compile(source, 'consistency-runner-test', 'exec'), namespace)
    assert len(calls) == 7
    assert len(list((tmp_path / 'experiment' / 'attempts').glob('*/failed-*.json'))) == 1
    exec(compile(source, 'consistency-runner-test', 'exec'), namespace)
    assert len(calls) == 7
    assert namespace['generation_counts']['resumed_results'] == 6
    assert all(path.stat().st_mode & 0o777 == 0o600
               for path in (tmp_path / 'experiment' / 'results').glob('*.json'))
    namespace['APPROVE_EXTERNAL_PROCESSING'] = False
    with pytest.raises(RuntimeError, match='Approve external'):
        exec(compile(source, 'consistency-runner-test', 'exec'), namespace)