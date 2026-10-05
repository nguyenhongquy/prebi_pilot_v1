import json

import pandas as pd
import pytest

from reflection_assessment_feedback.models import ReflectionDocument, ReflectionSegment
from reflection_assessment_feedback.segment_classification_gemini import GeminiReflectionPrediction
from reflection_assessment_feedback.segment_classification_hybrid import (
    HYBRID_CORRECTION_PROMPT,
    SegmentClassifierHybrid,
)
from reflection_assessment_feedback.segment_model_comparison import compare_models


class FakeGBERT:
    def __init__(self):
        self.calls = []

    def predict(self, texts, *, gold_in_scope=False):
        self.calls.append((texts, gold_in_scope))
        return pd.DataFrame([
            {"scope": 1, "SW": "1", "UA": "0", "HA": "0"} for text in texts
        ])


class FakeGemini:
    def __init__(self):
        self.prompts = []

    def with_structured_output(self, schema):
        assert schema is GeminiReflectionPrediction
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return {"predictions": [
            {"index": index, "scope": "in_scope",
             "component_bands": {"SW": "0", "UA": "2", "HA": "1"}}
            for index in range(2)
        ]}


def document():
    return ReflectionDocument(document_id="private-doc", segments=[
        ReflectionSegment(segment_id="s1", text="Opening reflection", order=0),
        ReflectionSegment(segment_id="s2", text="Later context", order=1),
    ])


def test_hybrid_corrects_dimension_presence_and_bands_with_full_context():
    baseline = FakeGBERT()
    model = FakeGemini()
    hybrid = SegmentClassifierHybrid(model, baseline)

    result = hybrid.predict_document(document())

    assert baseline.calls == [(["Opening reflection", "Later context"], False)]
    assert result.segments[0].component_bands == {"SW": "0", "UA": "2", "HA": "1"}
    prompt = model.prompts[0]
    assert "Opening reflection" in prompt and "Later context" in prompt
    assert "Bewertungsrubrik:" in prompt
    assert HYBRID_CORRECTION_PROMPT in prompt
    assert "Ausgangspunkt, kein Referenzlabel" in prompt
    assert "UA 1 vs. UA 2" in prompt
    assert "keine unnötige Korrektur erzwingen" in prompt
    assert "Hybrider Korrekturauftrag" in prompt
    automatic = json.loads(prompt.split("Vorlaeufige automatische GBERT-Analyse:\n")[1])
    assert automatic[0]["component_bands"] == {"SW": "1", "UA": "0", "HA": "0"}
    assert "private-doc" not in prompt


def test_hybrid_oracle_routes_only_constrained_segments_through_gbert_oracle():
    baseline = FakeGBERT()
    hybrid = SegmentClassifierHybrid(FakeGemini(), baseline)

    hybrid.predict_document(document(), gold_in_scope_segment_ids={"s2"})

    assert baseline.calls[1] == (["Later context"], True)


def test_hybrid_uses_shared_four_model_candidate_comparison():
    rows = pd.DataFrame([
        {"candidate_id": str(index), "document_id": "d", "segment_id": f"s{index}",
         "segment_order": index, "text": f"Reflection {index}", "scope_target": 1,
         "target_situationserfassung": 0, "target_analyse": 2,
         "target_konsequenzen": 1, "split": "test"}
        for index in range(2)
    ])
    baseline = FakeGBERT()
    model = FakeGemini()
    hybrid = SegmentClassifierHybrid(model, baseline)
    result = compare_models(rows, {
        "ML": baseline, "GBERT": baseline,
        "Gemini": SegmentClassifierHybrid(FakeGemini(), baseline), "Hybrid": hybrid,
    })

    assert set(result["summary"].model) == {"ML", "GBERT", "Gemini", "Hybrid"}
    assert result["summary"].query("model == 'Hybrid'").accuracy.eq(1).all()
    assert len(model.prompts) == 2


def test_hybrid_rejects_missing_baseline_predictions_before_provider_call():
    baseline = FakeGBERT()
    baseline.predict = lambda texts, gold_in_scope=False: pd.DataFrame()
    model = FakeGemini()

    with pytest.raises(ValueError, match="every reflection segment"):
        SegmentClassifierHybrid(model, baseline).predict_document(document())
    assert not model.prompts