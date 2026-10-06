from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import evaluate_g2_direct_gbert as evaluation


def test_generation_retries_invalid_evidence_before_accepting_valid_output(tmp_path, monkeypatch):
    output = object()
    generate = Mock(side_effect=[ValueError("Unknown evidence ID"), output])
    reserve = Mock()
    monkeypatch.setattr(evaluation, "reserve_request_slot", reserve)
    request = object()

    result = evaluation.generate_validated(SimpleNamespace(generate=generate), request, 1, tmp_path / "state", 7.5)

    assert result is output
    assert generate.call_count == 2
    assert reserve.call_count == 2
    generate.assert_called_with(request, repetition=1)


def test_generation_does_not_accept_persistently_invalid_output(tmp_path, monkeypatch):
    generate = Mock(side_effect=ValueError("Unknown evidence ID"))
    monkeypatch.setattr(evaluation, "reserve_request_slot", Mock())
    with pytest.raises(ValueError, match="Unknown evidence ID"):
        evaluation.generate_validated(SimpleNamespace(generate=generate), object(), 1, tmp_path / "state", 7.5)
    assert generate.call_count == 3