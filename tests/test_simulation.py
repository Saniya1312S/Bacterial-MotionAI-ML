"""Tests for the simulator."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.simulation import simulate_bacterium, STEPS


def test_trajectory_length():
    xs, ys, ths = simulate_bacterium(W=20, seed=0)
    assert len(xs) == len(ys) == len(ths) == STEPS + 1


def test_y_stays_inside_channel():
    for W in (10, 50):
        _, ys, _ = simulate_bacterium(W=W, hydro=True, chemical="attractant", seed=1)
        assert ys.min() >= 0 and ys.max() <= W


def test_same_seed_gives_same_trajectory():
    a = simulate_bacterium(W=20, chemical="repellent", seed=99)
    b = simulate_bacterium(W=20, chemical="repellent", seed=99)
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


def test_different_seeds_give_different_trajectories():
    a = simulate_bacterium(W=20, seed=1)
    b = simulate_bacterium(W=20, seed=2)
    assert not np.allclose(a[0], b[0])


def test_antibiotic_is_slower_on_average():
    """Averaged over many cells, antibiotic must reduce speed (single cells can overlap)."""
    def avg_speed(chem):
        v = []
        for s in range(30):
            xs, ys, _ = simulate_bacterium(W=50, chemical=chem, seed=s)
            v.append(np.hypot(np.diff(xs), np.diff(ys)).mean() / 0.01)
        return np.mean(v)
    assert avg_speed("antibiotic") < 0.7 * avg_speed("none")
