from __future__ import annotations

import gc
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from reflection_assessment_feedback.data import TARGET_COLUMNS


def score_predictions(
    gold: pd.DataFrame,
    predictions: pd.DataFrame,
    gold_scope_predictions: pd.DataFrame | None = None,
) -> dict:
    if gold.empty or len(gold) != len(predictions):
        raise ValueError("Predictions must cover every gold candidate in the same order.")
    scope = pd.to_numeric(gold["scope_target"], errors="raise").to_numpy()
    if not np.isin(scope, [0, 1]).all():
        raise ValueError("Gold scope must use known 0/1 labels.")
    in_scope = scope == 1
    if gold_scope_predictions is not None and len(gold_scope_predictions) != int(in_scope.sum()):
        raise ValueError("Gold-scope predictions must cover all gold in-scope candidates.")
    if not predictions["scope"].isin([0, 1]).all():
        raise ValueError("Predicted scope must use 0/1 labels.")
    summary = []
    per_class = []
    matrices = {}

    def add_task(task, expected, predicted, labels):
        if not len(expected):
            raise ValueError("Each comparison task requires gold examples.")
        precision, recall, f1, support = precision_recall_fscore_support(
            expected, predicted, labels=labels, zero_division=0,
        )
        summary.append({
            "task": task, "accuracy": float(accuracy_score(expected, predicted)),
            "macro_f1": float(f1.mean()),
            "supported_macro_f1": float(f1[support > 0].mean()),
            "weighted_f1": float(np.average(f1, weights=support)), "n": len(expected),
        })
        per_class.extend({
            "task": task, "class": str(label), "precision": float(class_precision),
            "recall": float(class_recall), "f1": float(class_f1), "support": int(class_support),
        } for label, class_precision, class_recall, class_f1, class_support in zip(
            labels, precision, recall, f1, support, strict=True,
        ))
        matrices[task] = {"labels": list(labels), "matrix": confusion_matrix(expected, predicted, labels=labels)}

    add_task("scope", scope, predictions["scope"].to_numpy(), [0, 1])
    for dimension, column in TARGET_COLUMNS.items():
        bands = pd.to_numeric(gold.loc[in_scope, column], errors="raise").to_numpy()
        if not np.isin(bands, [0, 1, 2, 3]).all():
            raise ValueError("Gold skill labels must be known bands 0-3.")
        labels = predictions[dimension].to_numpy()
        predicted_in_scope = predictions["scope"].to_numpy() == 1
        if not np.isin(labels[predicted_in_scope], ["0", "1", "2", "3"]).all():
            raise ValueError("In-scope predictions must use bands 0-3.")
        if not np.all(labels[~predicted_in_scope] == "N"):
            raise ValueError("Out-of-scope predictions must use N, not skill absence.")
        if gold_scope_predictions is not None:
            oracle = gold_scope_predictions[dimension].to_numpy()
            if not np.isin(oracle, ["0", "1", "2", "3"]).all():
                raise ValueError("Gold-scope predictions must use bands 0-3.")
            add_task(f"{dimension}_gold_scope", bands.astype(int).astype(str), oracle, ["0", "1", "2", "3"])
        expected = np.full(len(gold), "N", dtype=object)
        expected[in_scope] = bands.astype(int).astype(str)
        add_task(f"{dimension}_end_to_end", expected, labels, ["N", "0", "1", "2", "3"])
    return {
        "summary": pd.DataFrame(summary), "per_class": pd.DataFrame(per_class),
        "confusion_matrices": matrices,
    }


class GBERTPredictor:
    def __init__(self, checkpoint_directory: str | Path, settings: dict, *, device: str = "cpu") -> None:
        import torch
        from transformers import AutoConfig, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.settings = settings
        self.checkpoint_directory = Path(checkpoint_directory)
        self.checkpoints = {}
        for run_name, count in {"scope": 2, **{f"direct-{column.removeprefix('target_')}": 4 for column in TARGET_COLUMNS.values()}}.items():
            checkpoint = self.checkpoint_directory / f"{run_name}.pt"
            metadata_path = checkpoint.with_suffix(".json")
            if not checkpoint.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(f"Missing completed GBERT run: {run_name}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("num_labels") != count or not metadata.get("history") or Path(metadata.get("checkpoint", "")).name != checkpoint.name:
                raise ValueError(f"Incompatible GBERT metadata: {run_name}")
            self.checkpoints[run_name] = checkpoint
        revision = settings.get("model_revision") or None
        self.base_config = AutoConfig.from_pretrained(settings["model_name"], revision=revision, local_files_only=True)
        if self.base_config.model_type != "bert":
            raise ValueError("The saved GBERT runs require a BERT configuration.")
        self.tokenizer = AutoTokenizer.from_pretrained(settings["model_name"], revision=revision, local_files_only=True)

    def _predict_head(self, run_name: str, texts: list[str], count: int) -> np.ndarray:
        from copy import deepcopy
        from transformers import AutoModelForSequenceClassification

        config = deepcopy(self.base_config)
        config.num_labels = count
        model = AutoModelForSequenceClassification.from_config(config)
        model.load_state_dict(self.torch.load(self.checkpoints[run_name], map_location="cpu", weights_only=True), strict=True)
        model = model.to(self.device).eval()
        batches = []
        try:
            with self.torch.inference_mode():
                for start in range(0, len(texts), self.settings["batch_size"]):
                    encoded = self.tokenizer(
                        texts[start:start + self.settings["batch_size"]], truncation=True,
                        max_length=self.settings["max_length"], padding=True, return_tensors="pt",
                    )
                    logits = model(**{key: value.to(self.device) for key, value in encoded.items()}).logits
                    batches.append(logits.argmax(dim=-1).cpu().numpy())
        finally:
            del model
            gc.collect()
            if self.device.type == "mps":
                self.torch.mps.empty_cache()
            elif self.device.type == "cuda":
                self.torch.cuda.empty_cache()
        return np.concatenate(batches) if batches else np.array([], dtype=int)

    def predict(self, texts: list[str], *, gold_in_scope: bool = False) -> pd.DataFrame:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("GBERT prediction requires nonempty, nonblank texts.")
        unique = list(dict.fromkeys(texts))
        text_indices = {text: index for index, text in enumerate(unique)}
        inverse = np.array([text_indices[text] for text in texts])
        scope = np.ones(len(unique), dtype=int) if gold_in_scope else self._predict_head("scope", unique, 2)
        active = scope == 1
        result = pd.DataFrame({"scope": scope})
        for dimension, column in TARGET_COLUMNS.items():
            labels = np.full(len(unique), "N", dtype=object)
            if active.any():
                labels[active] = self._predict_head(
                    f"direct-{column.removeprefix('target_')}",
                    [text for text, selected in zip(unique, active, strict=True) if selected], 4,
                ).astype(str)
            result[dimension] = labels
        return result.iloc[inverse].reset_index(drop=True)


def compare_models(gold: pd.DataFrame, models: dict) -> dict:
    if "split" not in gold or not gold["split"].eq("test").all():
        raise ValueError("Final comparison requires the held-out test split.")
    if gold["candidate_id"].duplicated().any():
        raise ValueError("Test candidate IDs must be unique; preserve annotation bundles.")
    if gold["text"].isna().any() or not gold["text"].astype(str).str.strip().ne("").all():
        raise ValueError("Test text must not be missing or blank.")
    if len(models) < 2:
        raise ValueError("Comparison requires at least two models.")
    summaries = []
    per_class = []
    timings = []
    matrices = {}
    gold_mask = gold["scope_target"].eq(1)
    for name, model in models.items():
        started = perf_counter()
        predict_frame = getattr(model, "predict_frame", None)
        if callable(predict_frame):
            predictions = predict_frame(gold)
        else:
            predictions = model.predict(gold["text"].tolist())
        elapsed = perf_counter() - started
        started = perf_counter()
        if callable(predict_frame):
            oracle = predict_frame(gold, gold_in_scope=True).loc[gold_mask].reset_index(drop=True)
        else:
            oracle = model.predict(gold.loc[gold_mask, "text"].tolist(), gold_in_scope=True)
        oracle_elapsed = perf_counter() - started
        metrics = score_predictions(gold, predictions, oracle)
        summaries.append(metrics["summary"].assign(model=name))
        per_class.append(metrics["per_class"].assign(model=name))
        matrices[name] = metrics["confusion_matrices"]
        timings.append({
            "model": name, "end_to_end_seconds": elapsed,
            "gold_scope_diagnostic_seconds": oracle_elapsed,
            "milliseconds_per_candidate": elapsed / len(gold) * 1000,
        })
    return {
        "summary": pd.concat(summaries, ignore_index=True),
        "per_class": pd.concat(per_class, ignore_index=True),
        "timings": pd.DataFrame(timings), "confusion_matrices": matrices,
        "candidate_rows": len(gold), "unique_segments": gold["segment_id"].nunique(),
        "documents": gold["document_id"].nunique(),
    }