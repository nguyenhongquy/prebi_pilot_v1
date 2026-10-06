import pandas as pd
import pytest

from reflection_assessment_feedback.generation_consistency import consistency_tables, paired_consistency_differences


def test_four_variants_keep_historical_and_new_g2_separate():
    conditions = ("G1 historical", "G2 historical", "G3 historical", "G2 new direct GBERT")
    units = pd.DataFrame([
        {"document_id": "first", "mode": "test", "author_id": "author-one"},
        {"document_id": "second", "mode": "test", "author_id": "author-two"},
    ])
    rows = []
    for document_id in units.document_id:
        for condition in conditions:
            for dimension in ("SW", "UA", "HA"):
                for repetition in range(1, 6):
                    score = 1.0 if condition == "G2 historical" and repetition < 5 else 2.0
                    rows.append({"document_id": document_id, "condition": condition, "dimension_id": dimension, "repetition": repetition, "score": score})
    tables = consistency_tables(pd.DataFrame(rows), units=units, conditions=conditions)
    assert len(tables["within"]) == 12
    assert len(tables["between"]) == 18
    assert tables["within"].loc[tables["within"].condition.eq("G2 new direct GBERT"), "pairwise_exact"].eq(1.0).all()
    differences = paired_consistency_differences(
        tables["per_document"], samples=10,
        contrasts=(("G2 historical", "G2 new direct GBERT"),),
    )
    assert len(differences) == 3
    assert differences["difference"].eq(0.4).all()


def test_variant_names_must_be_unique():
    with pytest.raises(ValueError, match="unique condition"):
        consistency_tables(pd.DataFrame(), units=pd.DataFrame(), conditions=("G2", "G2"))