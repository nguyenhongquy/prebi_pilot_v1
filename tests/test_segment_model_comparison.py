import numpy as np
import pandas as pd
import pytest

from reflection_assessment_feedback.segment_model_comparison import GBERTPredictor, compare_models, score_predictions


def gold_rows():
    return pd.DataFrame({
        "candidate_id": ["a", "b", "c"], "segment_id": ["one", "one", "two"],
        "document_id": ["doc"] * 3, "split": ["test"] * 3,
        "text": ["Same segment text", "Same segment text", "Other segment text"],
        "scope_target": [1, 1, 0], "target_situationserfassung": [1, 2, np.nan],
        "target_analyse": [0, 0, np.nan], "target_konsequenzen": [0, 0, np.nan],
    })


class FixedPredictor:
    def predict(self, texts, *, gold_in_scope=False):
        if gold_in_scope:
            return pd.DataFrame({"scope": [1, 1], "SW": ["1", "1"], "UA": ["0", "0"], "HA": ["0", "0"]})
        return pd.DataFrame({"scope": [1, 1, 0], "SW": ["1", "1", "N"], "UA": ["0", "0", "N"], "HA": ["0", "0", "N"]})


def test_identical_models_use_identical_rows_and_metrics():
    comparison = compare_models(gold_rows(), {"ML": FixedPredictor(), "GBERT": FixedPredictor()})
    assert comparison["candidate_rows"] == 3
    assert comparison["unique_segments"] == 2
    summaries = comparison["summary"]
    pd.testing.assert_frame_equal(
        summaries.loc[summaries.model == "ML"].drop(columns="model").reset_index(drop=True),
        summaries.loc[summaries.model == "GBERT"].drop(columns="model").reset_index(drop=True),
    )
    assert summaries.loc[summaries.task == "SW_end_to_end", "n"].eq(3).all()
    assert summaries.loc[summaries.task == "SW_gold_scope", "accuracy"].eq(0.5).all()
    assert comparison["per_class"].query("task == 'SW_gold_scope' and `class` == '3'").support.eq(0).all()


def test_missing_predictions_and_scope_mismatch_rejected():
    predictor = FixedPredictor()
    predictions = predictor.predict([])
    oracle = predictor.predict([], gold_in_scope=True)
    with pytest.raises(ValueError, match="every gold candidate"):
        score_predictions(gold_rows(), predictions.iloc[:2], oracle)
    predictions.loc[2, "SW"] = "0"
    with pytest.raises(ValueError, match="must use N"):
        score_predictions(gold_rows(), predictions, oracle)
    with pytest.raises(ValueError, match="held-out test"):
        compare_models(gold_rows().assign(split="dev"), {"ML": predictor, "GBERT": predictor})


def test_gbert_predictor_routes_unique_texts_and_expands_candidates():
    predictor = object.__new__(GBERTPredictor)
    calls = []

    def predict_head(run_name, texts, count):
        calls.append((run_name, texts, count))
        return np.array([1, 0]) if run_name == "scope" else np.ones(len(texts), dtype=int)

    predictor._predict_head = predict_head
    result = predictor.predict(gold_rows().text.tolist())
    assert result.scope.tolist() == [1, 1, 0]
    assert result.SW.tolist() == ["1", "1", "N"]
    assert calls[0][1] == ["Same segment text", "Other segment text"]
    assert all(texts == ["Same segment text"] for _, texts, _ in calls[1:])
    calls.clear()
    oracle = predictor.predict(["Same segment text"] * 2, gold_in_scope=True)
    assert oracle.scope.tolist() == [1, 1]
    assert not any(name == "scope" for name, _, _ in calls)