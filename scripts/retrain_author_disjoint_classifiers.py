from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from reflection_assessment_feedback.experiment_config import load_experiment_config


def validate_splits() -> dict:
    config = load_experiment_config()
    directory = PROJECT_ROOT / config["paths"]["provisional_split_directory"]
    provenance = json.loads((directory / "author_disjoint_provenance.json").read_text())
    frames = []
    for split in ("train", "dev", "test"):
        for view in ("modeling_wide", "scope"):
            path = directory / f"provisional_{split}_{view}.csv"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["split_sha256"][path.name]
        frame = pd.read_csv(directory / f"provisional_{split}_modeling_wide.csv")
        assert frame["split"].eq(split).all()
        assert frame["text"].fillna("").str.strip().str.len().ge(10).all()
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    assert combined["author_id"].nunique() == 62
    for column in ("author_id", "document_id", "segment_id"):
        assert combined.groupby(column)["split"].nunique().eq(1).all(), column
    print("Verified saved author-disjoint splits: 62 authors; minimum 10 trimmed characters.", flush=True)
    return provenance


def execute_cell(notebook: dict, index: int, namespace: dict) -> None:
    cell = notebook["cells"][index]
    if cell["cell_type"] != "code":
        raise ValueError(f"Expected a code cell at index {index}")
    print(f"Executing training cell {index}", flush=True)
    exec(compile("".join(cell["source"]), f"training-notebook:cell-{index}", "exec"), namespace)


def train_ml(provenance: dict) -> None:
    notebook = json.loads((PROJECT_ROOT / "00_segment_classification_ml.ipynb").read_text())
    namespace: dict[str, Any] = {"__name__": "__ml_retraining__"}
    execute_cell(notebook, 1, namespace)
    namespace.update(RUN_TRAINING=True, RUN_TEST_EVALUATION=True, SAVE_MODEL=True)
    for index in (3, 5, 6, 8):
        execute_cell(notebook, index, namespace)
    model_path = namespace["model_path"]
    summary = {
        "model_path": str(model_path),
        "model_sha256": namespace["model_sha256"],
        "settings": namespace["asdict"](namespace["best_model"].settings),
        "dev_metrics": namespace["best_model"].evaluate(namespace["split_rows"]["dev"]),
        "test_metrics": namespace["test_metrics"],
        "split_provenance": provenance,
        "selection_split": "dev",
        "test_used_for_training": False,
    }
    timestamp = datetime.now(timezone.utc).strftime("author-disjoint-%Y%m%dT%H%M%S%fZ")
    directory = PROJECT_ROOT / "artifacts/segment-classification-ml" / timestamp
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "run-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(namespace["results"]).sort_values("dev_macro_f1", ascending=False).to_csv(
        directory / "dev-comparison.csv", index=False,
    )
    restored = namespace["SegmentClassifierML"].load(model_path)
    assert restored.evaluate(namespace["split_rows"]["test"]) == namespace["test_metrics"]
    print("ML model saved and reload verified:", model_path, flush=True)
    print("ML metrics:", directory, flush=True)


def train_gbert(provenance: dict, resume_directory: Path | None) -> None:
    notebook = json.loads((PROJECT_ROOT / "01_segment_classification.ipynb").read_text())
    config = load_experiment_config()
    namespace: dict[str, Any] = {
        "__name__": "__gbert_retraining__",
        "SEED": int(config["classifier"]["seed"]),
        "display": lambda value: print(value.to_string(index=False) if isinstance(value, pd.DataFrame) else value),
    }
    for index in (1, 4, 5):
        execute_cell(notebook, index, namespace)
    namespace["tqdm"] = partial(namespace["tqdm"], disable=True)
    if resume_directory is not None:
        namespace["RUN_DIR"].rmdir()
        directory = resume_directory.resolve()
        if not directory.is_dir():
            raise ValueError("Resume directory must exist")
        namespace["RUN_DIR"] = directory
        namespace["RETRAIN_RUN_ID"] = directory.name
        def fit_or_load(train_frame, dev_frame, label_col, num_labels, run_name, tokenizer):
            metadata_path = directory / f"{run_name}.json"
            if metadata_path.is_file():
                result = json.loads(metadata_path.read_text())
                if result.get("split_sha256") != namespace["SPLIT_SHA256"]:
                    raise ValueError("Saved checkpoint belongs to different splits")
                assert result["num_labels"] == num_labels and Path(result["checkpoint"]).is_file()
                print("Reusing completed author-disjoint head:", run_name, flush=True)
                return result
            return namespace["fit_gbert_classifier"](train_frame, dev_frame, label_col, num_labels, run_name, tokenizer)
        namespace["fit_or_load_gbert_classifier"] = fit_or_load
    directory = namespace["RUN_DIR"]
    (directory / "author-disjoint-provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("GBERT checkpoint directory:", directory, flush=True)
    namespace["LABELS"] = {
        "scope": ["out of scope", "in scope"],
        **{skill: ["absent", "band A", "band B", "band C"] for skill in namespace["SKILLS"]},
    }
    for index in (6, 8, 9, 10, 11, 12):
        execute_cell(notebook, index, namespace)
    manifest_path = directory / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["canonical_authors"] = 62
    manifest["author_normalization"] = provenance["author_normalization"]
    manifest["test_evaluation_completed"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("GBERT training and final test evaluation complete:", directory, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("ml", "gbert"), required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume-directory", type=Path)
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    provenance = validate_splits()
    if args.validate_only:
        return
    if args.model == "ml":
        if not os.getenv("PREBI_DATA_ROOT"):
            raise ValueError("PREBI_DATA_ROOT must be set before ML training")
        train_ml(provenance)
    else:
        train_gbert(provenance, args.resume_directory)


if __name__ == "__main__":
    main()