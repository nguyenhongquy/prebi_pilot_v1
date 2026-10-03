from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from reflection_assessment_feedback.data import TARGET_COLUMNS
from reflection_assessment_feedback.models import (
    IntermediateAnalysis,
    ReflectionDocument,
    SegmentAnalysis,
)


@dataclass(frozen=True)
class MLSettings:
    architecture: Literal["direct", "hierarchical"] = "direct"
    estimator: Literal["logistic", "linear_svc"] = "logistic"
    regularization: float = 1.0
    class_weight: Literal["balanced"] | None = None
    minimum_text_characters: int = 10
    word_max_features: int = 40000
    char_max_features: int = 60000
    seed: int = 20260929

    def __post_init__(self) -> None:
        if self.architecture not in {"direct", "hierarchical"}:
            raise ValueError("Unknown cascade architecture.")
        if self.estimator not in {"logistic", "linear_svc"}:
            raise ValueError("Unknown estimator.")
        if self.regularization <= 0 or self.minimum_text_characters < 1:
            raise ValueError("Regularization must be positive and minimum length at least 1.")
        if self.word_max_features < 1 or self.char_max_features < 1:
            raise ValueError("Feature limits must be positive.")


def eligible_rows(rows: pd.DataFrame, minimum_text_characters: int = 10) -> pd.Series:
    if minimum_text_characters < 1:
        raise ValueError("Minimum text length must be at least 1.")
    texts = rows["text"].fillna("").astype(str).str.strip()
    eligible = texts.str.len().ge(minimum_text_characters) & texts.str.contains(r"\w", regex=True)
    if "text_quality_excluded" in rows:
        excluded = rows["text_quality_excluded"].astype(str).str.lower().isin({"true", "1", "yes"})
        eligible &= ~excluded
    return eligible


class SegmentClassifierML:
    def __init__(self, settings: MLSettings | None = None) -> None:
        self.settings = settings or MLSettings()
        self.features = FeatureUnion([
            ("word", TfidfVectorizer(
                ngram_range=(1, 2), sublinear_tf=True,
                max_features=self.settings.word_max_features,
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                max_features=self.settings.char_max_features,
            )),
        ])
        self.heads: dict[str, object] = {}
        self.training_metadata: dict = {}

    def _fit_head(self, features, targets: np.ndarray):
        if not len(targets):
            raise ValueError("A classifier head has no eligible training examples.")
        if len(np.unique(targets)) == 1:
            head = DummyClassifier(strategy="constant", constant=int(targets[0]))
        elif self.settings.estimator == "linear_svc":
            head = LinearSVC(
                C=self.settings.regularization, class_weight=self.settings.class_weight,
                random_state=self.settings.seed, max_iter=5000,
            )
        else:
            head = LogisticRegression(
                C=self.settings.regularization, class_weight=self.settings.class_weight,
                random_state=self.settings.seed, max_iter=1000, solver="lbfgs",
            )
        return head.fit(features, targets)

    def fit(self, training_rows: pd.DataFrame) -> SegmentClassifierML:
        required = {"text", "scope_target", *TARGET_COLUMNS.values()}
        if not required.issubset(training_rows.columns):
            raise ValueError(f"Missing training columns: {sorted(required - set(training_rows.columns))}")
        if "split" in training_rows and not training_rows["split"].eq("train").all():
            raise ValueError("Fit accepts training rows only.")
        rows = training_rows.loc[eligible_rows(
            training_rows, self.settings.minimum_text_characters,
        )].copy()
        if rows.empty:
            raise ValueError("No eligible training text remains.")
        scope = pd.to_numeric(rows["scope_target"], errors="raise")
        if not scope.isin([0, 1]).all():
            raise ValueError("Training scope labels must be known 0/1 values.")
        in_scope = scope.eq(1).to_numpy()
        for column in TARGET_COLUMNS.values():
            targets = pd.to_numeric(rows.loc[in_scope, column], errors="raise")
            if not targets.isin([0, 1, 2, 3]).all():
                raise ValueError("In-scope skill labels must be known bands 0-3.")
        features = self.features.fit_transform(rows["text"].astype(str))
        heads = {"scope": self._fit_head(features, scope.to_numpy(dtype=int))}
        for dimension, column in TARGET_COLUMNS.items():
            targets = rows.loc[in_scope, column].to_numpy(dtype=int)
            skill_features = features[in_scope]
            if self.settings.architecture == "direct":
                heads[dimension] = self._fit_head(skill_features, targets)
            else:
                present = targets > 0
                heads[f"{dimension}_presence"] = self._fit_head(skill_features, present.astype(int))
                heads[f"{dimension}_band"] = self._fit_head(skill_features[present], targets[present])
        self.heads = heads
        self.training_metadata = {
            "input_rows": len(training_rows), "eligible_rows": len(rows),
            "excluded_rows": len(training_rows) - len(rows),
            "in_scope_rows": int(in_scope.sum()), "feature_count": features.shape[1],
            "head_classes": {name: head.classes_.tolist() for name, head in heads.items()},
        }
        return self

    def predict(self, texts: list[str], *, gold_in_scope: bool = False) -> pd.DataFrame:
        if not self.heads:
            raise ValueError("Fit or load a classifier before prediction.")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Prediction requires nonblank string texts.")
        columns = ["scope", *TARGET_COLUMNS]
        if not texts:
            return pd.DataFrame(columns=columns)
        features = self.features.transform(texts)
        scope = np.ones(len(texts), dtype=int) if gold_in_scope else self.heads["scope"].predict(features)
        result = pd.DataFrame({"scope": scope})
        active = scope == 1
        for dimension in TARGET_COLUMNS:
            labels = np.full(len(texts), "N", dtype=object)
            if active.any():
                if self.settings.architecture == "direct":
                    labels[active] = self.heads[dimension].predict(features[active]).astype(str)
                else:
                    present = self.heads[f"{dimension}_presence"].predict(features[active]).astype(bool)
                    bands = np.zeros(int(active.sum()), dtype=int)
                    if present.any():
                        bands[present] = self.heads[f"{dimension}_band"].predict(features[active][present])
                    labels[active] = bands.astype(str)
            result[dimension] = labels
        return result

    def predict_document(self, document: ReflectionDocument) -> IntermediateAnalysis:
        predictions = self.predict([segment.text for segment in document.segments])
        return IntermediateAnalysis(
            source="predicted",
            segments=[
                SegmentAnalysis(
                    segment_id=segment.segment_id,
                    scope="in_scope" if int(predictions.iloc[index]["scope"]) else "out_of_scope",
                    component_bands={dimension: predictions.iloc[index][dimension] for dimension in TARGET_COLUMNS},
                )
                for index, segment in enumerate(document.segments)
            ],
        )

    def evaluate(self, rows: pd.DataFrame) -> dict:
        if rows.empty:
            raise ValueError("Evaluation requires at least one row.")
        scope = pd.to_numeric(rows["scope_target"], errors="raise")
        if not scope.isin([0, 1]).all():
            raise ValueError("Evaluation requires known scope labels.")
        predictions = self.predict(rows["text"].tolist())
        gold_mask = scope.eq(1).to_numpy()
        if not gold_mask.any():
            raise ValueError("Evaluation requires gold in-scope rows.")
        gold_predictions = self.predict(rows.loc[gold_mask, "text"].tolist(), gold_in_scope=True)
        reports = {"scope": classification_report(
            scope, predictions["scope"], labels=[0, 1], output_dict=True, zero_division=0,
        )}
        skill_scores = []
        for dimension, column in TARGET_COLUMNS.items():
            numeric_targets = pd.to_numeric(rows.loc[gold_mask, column], errors="raise")
            if not numeric_targets.isin([0, 1, 2, 3]).all():
                raise ValueError("Evaluation requires known in-scope skill bands 0-3.")
            gold = numeric_targets.astype(int).astype(str)
            reports[f"{dimension}_gold_scope"] = classification_report(
                gold, gold_predictions[dimension], labels=["0", "1", "2", "3"],
                output_dict=True, zero_division=0,
            )
            expected = np.full(len(rows), "N", dtype=object)
            expected[gold_mask] = gold.to_numpy()
            reports[f"{dimension}_end_to_end"] = classification_report(
                expected, predictions[dimension], labels=["N", "0", "1", "2", "3"],
                output_dict=True, zero_division=0,
            )
            supported = sorted(set(expected))
            skill_scores.append(f1_score(
                expected, predictions[dimension], labels=supported, average="macro", zero_division=0,
            ))
        return {
            "rows": len(rows), "reports": reports,
            "selection_macro_f1": float(np.mean(skill_scores)),
            "selection_policy": "Mean end-to-end macro-F1 across skills, over gold-supported labels",
        }

    def save(self, path: str | Path) -> Path:
        if not self.heads:
            raise ValueError("Cannot save an unfitted classifier.")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "format_version": 1, "sklearn_version": sklearn.__version__,
            "settings": asdict(self.settings), "classifier": self,
        }, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> SegmentClassifierML:
        package = joblib.load(path)
        if package.get("format_version") != 1 or package.get("sklearn_version") != sklearn.__version__:
            raise ValueError("Model format or scikit-learn version mismatch; retrain in this environment.")
        classifier = package.get("classifier")
        if not isinstance(classifier, cls) or not classifier.heads:
            raise ValueError("Artifact does not contain a fitted ML classifier.")
        return classifier


def export_g2_prediction(
    classifier: SegmentClassifierML,
    document: ReflectionDocument,
    *,
    protected_root: str | Path,
    project_root: str | Path,
    config_sha256: str,
    model_sha256: str,
) -> Path:
    from urllib.parse import quote

    root = Path(protected_root).expanduser().resolve()
    source = Path(project_root).resolve()
    if root == source or source in root.parents:
        raise ValueError("Protected storage must be outside the project source tree.")
    if len(model_sha256) != 64 or any(character not in "0123456789abcdef" for character in model_sha256):
        raise ValueError("Model SHA-256 must be a lowercase hexadecimal digest.")
    analysis = classifier.predict_document(document)
    package = {
        "document_id": document.document_id,
        "experiment_config_sha256": config_sha256,
        "classifier_backend": "tfidf",
        "classifier_model_sha256": model_sha256,
        "classifier_settings": asdict(classifier.settings),
        "document_sha256": hashlib.sha256(document.model_dump_json().encode()).hexdigest(),
        "analysis": analysis.model_dump(mode="json"),
    }
    directory = root / "g2" / "predictions" / "tfidf" / model_sha256
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"predicted-analysis-{quote(document.document_id, safe='')}.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != package:
            raise ValueError("Refusing to overwrite a different prediction artifact.")
        return path
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, suffix=".tmp", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(package, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path