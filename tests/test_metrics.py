import numpy as np
import pytest

from cubesat.metrics import overshoot_pct, rms_after, settling_time


def test_settling_time_known_signal():
    t = np.linspace(0, 10, 1001)
    err = np.where(t < 4.0, 1.0, 0.0)
    assert settling_time(t, err, 0.1, 10.0) == pytest.approx(4.0, abs=0.011)


def test_settling_never_settles_returns_horizon():
    t = np.linspace(0, 10, 101)
    assert settling_time(t, np.ones_like(t), 0.1, 10.0) == 10.0


def test_settling_always_inside_is_zero():
    t = np.linspace(0, 10, 101)
    assert settling_time(t, np.zeros_like(t), 0.1, 10.0) == 0.0


def test_overshoot():
    y = np.array([0, 0.5, 1.1, 1.0])
    assert overshoot_pct(y, 1.0) == pytest.approx(10.0)
    assert overshoot_pct(-y, -1.0) == pytest.approx(10.0)
    assert overshoot_pct(np.array([0, 0.9, 1.0]), 1.0) == 0.0


def test_rms_after():
    t = np.arange(10.0)
    e = np.where(t < 5, 100.0, 2.0)
    assert rms_after(t, e, 5.0) == pytest.approx(2.0)
