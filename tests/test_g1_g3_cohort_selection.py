import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

import reflection_assessment_feedback.data as reflection_data
import reflection_assessment_feedback.development_cohorts as cohorts
from reflection_assessment_feedback.experiment_config import load_experiment_config


NOTEBOOK = Path(__file__).resolve().parents[1] / "03_generate_g1_g3.ipynb"


def sources():
    return [
        "".join(cell["source"])
        for cell in json.loads(NOTEBOOK.read_text())["cells"]
        if cell["cell_type"] == "code"
    ]


@pytest.mark.parametrize("selection", ["playground", "remaining-test", None])
def test_g1_g3_loading_selects_one_shared_cohort(monkeypatch, selection):
    config = load_experiment_config()
    playground_ids = ["synthetic-188", "synthetic-126"]
    remaining_ids = [f"synthetic-test-{i}" for i in range(config["dataset"]["expected_generation_document_count"])]

    def feedback(ids):
        return pd.DataFrame({"document_id": ids, "feedback_text": ["Synthetic feedback"] * len(ids)})

    playground = Mock(return_value=feedback(playground_ids))
    remaining = Mock(return_value=feedback(remaining_ids))
    monkeypatch.setattr(cohorts, "load_playground_feedback_cohort", playground)
    monkeypatch.setattr(cohorts, "load_remaining_test_feedback_cohort", remaining)
    human = SimpleNamespace(segments=[SimpleNamespace(candidates=["synthetic-candidate"])])
    loader = Mock(side_effect=lambda path, identifier: (
        SimpleNamespace(document_id=identifier, segments=["synthetic-passage"]), human,
    ))
    monkeypatch.setattr(reflection_data, "load_human_analysis_context", loader)
    monkeypatch.setattr(importlib, "reload", lambda module: module)
    namespace = {} if selection is None else {"GENERATION_COHORT": selection}
    source = next(source for source in sources() if "DEVELOPMENT_BATCH = []" in source)
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)

    expected = playground_ids if selection == "playground" else remaining_ids
    assert [item["document"].document_id for item in namespace["DEVELOPMENT_BATCH"]] == expected
    assert all(item["human_context"] is human for item in namespace["DEVELOPMENT_BATCH"])
    assert [call.args[1] for call in loader.call_args_list] == expected
    assert namespace["EXPECTED_DOCUMENT_COUNT"] == len(expected)
    assert namespace["DEVELOPMENT_COHORT"] == (
        "playground-author-188-v1" if selection == "playground" else config["dataset"]["generation_cohort"]
    )
    assert playground.call_count == (1 if selection == "playground" else 0)
    assert remaining.call_count == (0 if selection == "playground" else 1)

    validation = next(source for source in sources() if "if len(SAVED_COMPARISONS)" in source)
    count_guard = validation.split("items_by_id =", 1)[0]
    namespace["SAVED_COMPARISONS"] = [None] * len(expected)
    exec(count_guard, namespace)
    namespace["SAVED_COMPARISONS"] = [None] * (len(expected) - 1)
    with pytest.raises(ValueError, match="Not every"):
        exec(count_guard, namespace)


def test_g1_g3_invalid_cohort_fails_before_loading_documents(monkeypatch):
    loader = Mock()
    monkeypatch.setattr(reflection_data, "load_human_analysis_context", loader)
    monkeypatch.setattr(importlib, "reload", lambda module: module)
    source = next(source for source in sources() if "DEVELOPMENT_BATCH = []" in source)
    with pytest.raises(ValueError, match="GENERATION_COHORT"):
        exec(compile(source, str(NOTEBOOK), "exec"), {"GENERATION_COHORT": "invalid"})
    loader.assert_not_called()
