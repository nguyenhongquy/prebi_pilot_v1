import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

import reflection_assessment_feedback.development_cohorts as cohorts
from reflection_assessment_feedback.experiment_config import load_experiment_config


NOTEBOOK = Path(__file__).resolve().parents[1] / "04_generate_g2.ipynb"


def sources():
    return {
        cell["id"]: "".join(cell["source"])
        for cell in json.loads(NOTEBOOK.read_text())["cells"]
        if cell["cell_type"] == "code"
    }


@pytest.mark.parametrize("selection", ["playground", "remaining-test", None])
def test_g2_cohort_loading_and_tracing_guards(monkeypatch, selection):
    config = load_experiment_config()
    playground_ids = ["synthetic-188", "synthetic-126"]
    remaining_ids = [
        f"synthetic-test-{i}"
        for i in range(config["dataset"]["expected_generation_document_count"])
    ]
    playground = Mock(return_value=pd.DataFrame({"document_id": playground_ids}))
    remaining = Mock(return_value=pd.DataFrame({"document_id": remaining_ids}))
    monkeypatch.setattr(cohorts, "load_playground_feedback_cohort", playground)
    monkeypatch.setattr(cohorts, "load_remaining_test_feedback_cohort", remaining)
    expected = playground_ids if selection == "playground" else remaining_ids
    rows = pd.DataFrame([
        {"document_id": identifier, "segment_id": f"{identifier}-1",
         "text": "Synthetic reflection passage.", "segment_order": 1}
        for identifier in playground_ids + remaining_ids
    ])
    read_csv = Mock(return_value=rows)
    monkeypatch.setattr(pd, "read_csv", read_csv)
    namespace = {} if selection is None else {"GENERATION_COHORT": selection}
    cells = sources()
    exec(cells["c0f8028d"], namespace)
    exec(compile(cells["6a6d9267"], str(NOTEBOOK), "exec"), namespace)
    assert list(namespace["DEVELOPMENT_DOCUMENTS"]) == sorted(expected)
    assert namespace["EXPECTED_DOCUMENT_COUNT"] == len(expected)
    assert namespace["DEVELOPMENT_COHORT"] == (
        "playground-author-188-v1" if selection == "playground"
        else config["dataset"]["generation_cohort"]
    )
    assert playground.call_count == (1 if selection == "playground" else 0)
    assert remaining.call_count == (0 if selection == "playground" else 1)
    assert read_csv.call_args.kwargs["usecols"] == [
        "document_id", "segment_id", "text", "segment_order",
    ]

    generation = cells["35a725d5"]
    guards = generation[
        generation.index("if GENERATION_COHORT != LOADED_GENERATION_COHORT:"):
        generation.index("PROTECTED_ROOT =")
    ]
    namespace.update({
        "GENERATION_CONFIG": config["generation"],
        "PREDICTED_ANALYSES": {identifier: None for identifier in expected},
        "os": SimpleNamespace(getenv=lambda name, default=None: (
            "https://eu.api.smith.langchain.com" if name == "LANGSMITH_ENDPOINT"
            else "synthetic"
        )),
    })
    with pytest.raises(AssertionError, match="Confirm provider processing"):
        exec(guards, namespace)
    namespace["APPROVE_EXTERNAL_PROCESSING"] = True
    exec(guards, namespace)
    assert namespace["ENABLE_LANGSMITH_TRACING"] is True
    assert namespace["EXPECTED_DOCUMENT_COUNT"] == len(expected)

    model = Mock()
    runner = Mock()
    namespace.update({"ChatGoogleGenerativeAI": model, "RubricGenerationRunner": runner})
    constructors = generation[
        generation.index("chat_model = ChatGoogleGenerativeAI("):
        generation.index("G2_RUN_RECORDS = []")
    ]
    exec(constructors, namespace)
    assert runner.call_args.kwargs["enable_langsmith_tracing"] is True
    assert "with tracing_context(enabled=ENABLE_LANGSMITH_TRACING):" in generation

    namespace["GENERATION_COHORT"] = (
        "remaining-test" if selection == "playground" else "playground"
    )
    with pytest.raises(ValueError, match="Cohort selection changed"):
        exec(guards, namespace)
    namespace["GENERATION_COHORT"] = namespace["LOADED_GENERATION_COHORT"]
    namespace["GENERATION_CONFIG"] = {**config["generation"], "enable_langsmith_tracing": False}
    with pytest.raises(AssertionError, match="Tracing is required"):
        exec(guards, namespace)


def test_invalid_g2_cohort_fails_before_loading(monkeypatch):
    loader = Mock()
    monkeypatch.setattr(pd, "read_csv", loader)
    namespace = {"GENERATION_COHORT": "invalid"}
    exec(sources()["c0f8028d"], namespace)
    with pytest.raises(ValueError, match="GENERATION_COHORT must be"):
        exec(sources()["6a6d9267"], namespace)
    loader.assert_not_called()


def test_all_g2_cells_compile():
    for identifier, source in sources().items():
        compile(source, f"{NOTEBOOK}:{identifier}", "exec")
