import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reflection_assessment_feedback.models import ReflectionDocument, ReflectionSegment
from reflection_assessment_feedback.segment_classification_ml import (
    MLSettings, SegmentClassifierML, eligible_rows, export_g2_prediction,
)


@pytest.fixture
def training_rows():
    rows = []
    for repeat in range(3):
        for band in range(4):
            rows.append({
                "text": f"Unterricht Reflexion Analyse Konsequenzen Stufe {band} Beispiel {repeat}",
                "scope_target": 1, "target_situationserfassung": band,
                "target_analyse": band, "target_konsequenzen": band, "split": "train",
            })
        rows.append({
            "text": f"Organisatorische Pause Wetter Mittagessen {repeat}", "scope_target": 0,
            "target_situationserfassung": np.nan, "target_analyse": np.nan,
            "target_konsequenzen": np.nan, "split": "train",
        })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("architecture", ["direct", "hierarchical"])
@pytest.mark.parametrize("estimator", ["logistic", "linear_svc"])
def test_cascades_and_roundtrip(training_rows, tmp_path, architecture, estimator):
    model = SegmentClassifierML(MLSettings(architecture=architecture, estimator=estimator)).fit(training_rows)
    assert len(model.heads) == (4 if architecture == "direct" else 7)
    predictions = model.predict(training_rows.text.tolist())
    assert set(predictions.loc[predictions.scope == 0, "SW"]) <= {"N"}
    assert set(predictions.loc[predictions.scope == 1, "SW"]) <= {"0", "1", "2", "3"}
    report = model.evaluate(training_rows)
    assert 0 <= report["selection_macro_f1"] <= 1
    assert report["reports"]["SW_gold_scope"]["3"]["support"] == 3
    loaded = SegmentClassifierML.load(model.save(tmp_path / "model.joblib"))
    pd.testing.assert_frame_equal(predictions, loaded.predict(training_rows.text.tolist()))
    document = ReflectionDocument(document_id="example", segments=[
        ReflectionSegment(segment_id="segment", text=training_rows.text.iloc[0], order=0),
    ])
    analysis = loaded.predict_document(document)
    assert analysis.source == "predicted"
    assert analysis.segments[0].segment_id == "segment"


def test_quality_filter_and_training_only(training_rows):
    rows = pd.DataFrame({
        "text": ["", "...", "kurz", "Langer Unterrichtstext", "Noch ein Unterrichtstext"],
        "text_quality_excluded": [False, False, False, True, False],
    })
    assert eligible_rows(rows, 10).tolist() == [False, False, False, False, True]
    with pytest.raises(ValueError, match="training rows only"):
        SegmentClassifierML().fit(training_rows.assign(split="dev"))


def test_constant_heads_and_empty_band_guard(training_rows):
    in_scope = training_rows.loc[training_rows.scope_target == 1].copy()
    for column in ["target_situationserfassung", "target_analyse", "target_konsequenzen"]:
        in_scope[column] = 2
    model = SegmentClassifierML(MLSettings(architecture="hierarchical")).fit(in_scope)
    assert set(model.predict(in_scope.text.tolist()).SW) == {"2"}
    in_scope["target_analyse"] = 0
    with pytest.raises(ValueError, match="no eligible training examples"):
        SegmentClassifierML(MLSettings(architecture="hierarchical")).fit(in_scope)


def test_fit_does_not_learn_evaluation_vocabulary(training_rows):
    model = SegmentClassifierML().fit(training_rows)
    vocabulary = dict(model.features.transformer_list[0][1].vocabulary_)
    model.predict(["exclusivelyheldouttoken"])
    assert model.features.transformer_list[0][1].vocabulary_ == vocabulary
    assert "exclusivelyheldouttoken" not in vocabulary


def test_protected_prediction_export(training_rows, tmp_path):
    model = SegmentClassifierML().fit(training_rows)
    document = ReflectionDocument(document_id="doc/one", segments=[
        ReflectionSegment(segment_id="segment", text=training_rows.text.iloc[0], order=0),
    ])
    options = {
        "protected_root": tmp_path / "protected", "project_root": tmp_path / "source",
        "config_sha256": "b" * 64, "model_sha256": "a" * 64,
    }
    path = export_g2_prediction(model, document, **options)
    assert path.name == "predicted-analysis-doc%2Fone.json"
    assert path.stat().st_mode & 0o777 == 0o600
    package = json.loads(path.read_text())
    assert package["analysis"]["source"] == "predicted"
    assert package["classifier_backend"] == "tfidf"
    assert export_g2_prediction(model, document, **options) == path
    with pytest.raises(ValueError, match="overwrite"):
        export_g2_prediction(model, document, **{**options, "config_sha256": "c" * 64})
    with pytest.raises(ValueError, match="outside"):
        export_g2_prediction(model, document, **{**options, "protected_root": tmp_path / "source" / "data"})


def test_g2_notebook_ml_backend(training_rows, tmp_path, monkeypatch):
    from reflection_assessment_feedback import EXPERT_RUBRIC
    from reflection_assessment_feedback.experiment_config import experiment_config_sha256

    root = Path(__file__).resolve().parents[1]
    model = SegmentClassifierML().fit(training_rows)
    protected = tmp_path / "protected"
    model_path = model.save(protected / "model.joblib")
    document = ReflectionDocument(document_id="example", segments=[
        ReflectionSegment(segment_id="segment", text=training_rows.text.iloc[0], order=0),
    ])
    notebook = json.loads((root / "04_generate_g2.ipynb").read_text())
    cell = next(
        cell for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "PREDICTION_BACKEND =" in "".join(cell["source"])
    )
    source = "".join(cell["source"])
    source = source.replace("PREDICTION_BACKEND = 'gbert'", "PREDICTION_BACKEND = 'tfidf'")
    source = source.replace("ML_MODEL_PATH = None", f"ML_MODEL_PATH = {str(model_path)!r}")
    source = source.replace("WRITE_ML_PREDICTIONS = False", "WRITE_ML_PREDICTIONS = True")
    monkeypatch.setenv("PREBI_DATA_ROOT", str(protected))
    namespace = {
        "Path": Path, "os": os, "WORKSPACE_DIR": root, "EXPERT_RUBRIC": EXPERT_RUBRIC,
        "DEVELOPMENT_DOCUMENTS": {document.document_id: document},
    }
    exec(compile(source, "g2-prediction-cell", "exec"), namespace)
    assert namespace["PREDICTED_ANALYSES"]["example"].source == "predicted"
    prediction_path = namespace["PREDICTION_PATHS"]["example"]
    assert json.loads(prediction_path.read_text())["experiment_config_sha256"] == experiment_config_sha256()
    exec(compile(source, "g2-prediction-cell", "exec"), namespace)
    tampered = json.loads(prediction_path.read_text())
    tampered["document_sha256"] = "wrong"
    prediction_path.write_text(json.dumps(tampered))
    read_only = source.replace("WRITE_ML_PREDICTIONS = True", "WRITE_ML_PREDICTIONS = False")
    with pytest.raises(ValueError, match="different document text"):
        exec(compile(read_only, "g2-prediction-cell", "exec"), namespace)