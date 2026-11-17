"""Unit tests for the accuracy-evaluation metrics (benchmarks/eval_accuracy.py)."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "eval_accuracy", Path(__file__).resolve().parents[1] / "benchmarks" / "eval_accuracy.py"
)
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)
A = ev.Attempt


def sample():
    return [
        A("genuine", "1", "1", 0.80, True),   # correct
        A("genuine", "1", "1", 0.55, True),   # correct at 0.50, rejected at 0.60
        A("genuine", "2", "1", 0.52, True),   # misidentified at 0.50
        A("genuine", "2", None, 0.0, False),  # no face
        A("impostor", None, "1", 0.51, True), # false accept at 0.50
        A("impostor", None, "2", 0.30, True),
        A("impostor", None, "1", 0.20, True),
        A("impostor", None, None, 0.0, False),
    ]


def test_rates_at_threshold_050():
    r = ev.rates_at(sample(), 0.50)
    assert r["genuine"] == 4 and r["impostor"] == 4
    assert r["TAR"] == pytest.approx(2 / 4)
    assert r["FNIR"] == pytest.approx(2 / 4)
    assert r["misID"] == pytest.approx(1 / 4)
    assert r["FPIR"] == pytest.approx(1 / 4)


def test_rates_at_threshold_060():
    r = ev.rates_at(sample(), 0.60)
    assert r["TAR"] == pytest.approx(1 / 4)
    assert r["misID"] == 0
    assert r["FPIR"] == 0


def test_tar_at_fpir_picks_threshold_above_allowed_impostors():
    attempts = [A("genuine", "1", "1", s, True) for s in (0.9, 0.7, 0.6, 0.4)]
    attempts += [A("impostor", None, "1", s / 100, True) for s in range(100)]  # 0.00 .. 0.99
    r = ev.tar_at_fpir(attempts, 0.01)  # allow 1 of 100 impostors
    assert r["FPIR"] <= 0.01
    assert r["threshold"] == pytest.approx(0.98, abs=1e-3)


def test_eer_none_without_impostors():
    assert ev.eer([A("genuine", "1", "1", 0.9, True)]) is None


def test_eer_balances_rates():
    attempts = [A("genuine", "1", "1", s, True) for s in (0.9, 0.8, 0.7, 0.45)]
    attempts += [A("impostor", None, "1", s, True) for s in (0.1, 0.2, 0.3, 0.6)]
    r = ev.eer(attempts)
    assert r["EER"] == pytest.approx(0.25)


def test_histogram_counts_all_scores():
    h = ev.histogram([0.05, 0.5, 0.55, 0.99])
    assert sum(h["counts"]) == 4 and len(h["edges"]) == 21
