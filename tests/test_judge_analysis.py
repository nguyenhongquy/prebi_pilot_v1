import pandas as pd
import pytest

from reflection_assessment_feedback.judge_analysis import NEW_G2_LABEL, compare_new_g2_scores


def synthetic_scores():
    historical = pd.DataFrame([
        {"phase": phase, "document_id": document_id, "task": "feedback_quality", "criterion_id": "correctness", "condition": condition, "score": score, "unable_to_judge": False}
        for phase, document_id in [("test", "one"), ("test", "two"), ("playground", "other")]
        for condition, score in [("G1", 1), ("G2", 2), ("G3", 3)]
    ])
    new = pd.DataFrame([
        {"phase": "test", "document_id": document_id, "task": "feedback_quality", "criterion_id": "correctness", "condition": NEW_G2_LABEL, "score": 3, "unable_to_judge": False}
        for document_id in ("one", "two")
    ])
    return historical, new


def test_new_g2_comparison_keeps_versions_separate_and_matches_test_documents():
    historical, new = synthetic_scores()
    summary, differences = compare_new_g2_scores(historical, new)
    assert len(summary) == 4
    assert summary["document_count"].eq(2).all()
    old_g2 = differences.loc[differences["comparison"].str.endswith("G2 historical")].iloc[0]
    assert old_g2["mean_difference"] == 1
    assert old_g2["improved"] == 2
    assert old_g2["paired_documents"] == 2


def test_new_g2_comparison_rejects_duplicate_repetitions():
    historical, new = synthetic_scores()
    with pytest.raises(ValueError, match="one rating"):
        compare_new_g2_scores(historical, pd.concat([new, new]))


def test_new_g2_comparison_reports_only_scorable_pairs():
    historical, new = synthetic_scores()
    new.loc[0, "score"] = None
    new.loc[0, "unable_to_judge"] = True
    summary, differences = compare_new_g2_scores(historical, new)
    assert differences["paired_documents"].eq(1).all()
    assert summary.loc[summary["condition"].eq(NEW_G2_LABEL), "unable_to_judge"].iloc[0] == 1