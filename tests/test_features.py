"""Tests for feature extraction."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.simulation import simulate_bacterium, STEPS, DT
from src.features import extract_features, FEATURE_COLS


def test_returns_all_features_without_nan():
    xs, ys, _ = simulate_bacterium(W=20, hydro=True, chemical="repellent", seed=2)
    feats = extract_features(xs, ys, W=20)
    assert list(feats.keys()) == FEATURE_COLS
    assert len(feats) == 11
    assert not any(np.isnan(v) for v in feats.values())


def test_speed_is_close_to_physical_value():
    """Speeds must be near V0 = 20 um/s (this catches the old periodic-boundary bug)."""
    xs, ys, _ = simulate_bacterium(W=50, chemical="none", seed=42)
    feats = extract_features(xs, ys, W=50)
    assert 10 < feats["mean_speed"] < 30
    assert feats["std_speed"] < 5


def test_straight_line_has_straightness_one():
    t = np.arange(STEPS + 1)
    xs, ys = 0.2 * t, np.full(STEPS + 1, 25.0)          # 20 um/s along x
    feats = extract_features(xs, ys, W=50)
    assert abs(feats["straightness"] - 1.0) < 1e-6
    assert abs(feats["mean_speed"] - 20.0) < 1e-6
    assert abs(feats["persistence_1s"] - 1.0) < 1e-6
    assert feats["reorient_count"] == 0


def test_circle_has_nonzero_turn_bias():
    """Moving in a circle always turns the same way -> turn_bias should be clearly non-zero."""
    t = np.arange(STEPS + 1) * 0.02
    xs, ys = 5 * np.cos(t) + 50, 5 * np.sin(t) + 50
    feats = extract_features(xs, ys, W=100)
    assert feats["turn_bias"] > 0.01
