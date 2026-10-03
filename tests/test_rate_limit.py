from pathlib import Path

import pytest

from reflection_assessment_feedback import rate_limit


def test_reserve_request_slot_enforces_interval_and_private_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_time = [100.0]

    def fake_sleep(seconds: float) -> None:
        current_time[0] += seconds

    monkeypatch.setattr(rate_limit.time, "time", lambda: current_time[0])
    monkeypatch.setattr(rate_limit.time, "sleep", fake_sleep)
    state_path = tmp_path / "private" / "gemini.state"

    assert rate_limit.reserve_request_slot(state_path, 7.5) == 0.0
    assert rate_limit.reserve_request_slot(state_path, 7.5) == 7.5
    assert rate_limit.reserve_request_slot(state_path, 7.5) == 7.5
    assert state_path.stat().st_mode & 0o777 == 0o600


def test_reserve_request_slot_rejects_nonpositive_interval(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        rate_limit.reserve_request_slot(tmp_path / "rate.state", 0)
