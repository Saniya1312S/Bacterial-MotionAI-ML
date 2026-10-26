"""
Feature extraction from a single trajectory.

Only the (x, y) positions are used -- exactly what a microscope tracker gives you.
(We do NOT use the hidden orientation angle, because it cannot be observed in real data.)

11 features in 3 groups:
    Speed        : mean_speed, std_speed, cv_speed
    Path shape   : straightness, mean_turn_angle, turn_bias, persistence_1s, reorient_count
    Wall contact : mean_alignment, near_wall_frac, wall_contact_frac
"""
import numpy as np
from typing import Dict

from .simulation import DT  # seconds per step (shared with the simulation)

FEATURE_COLS = [
    "mean_speed",
    "std_speed",
    "cv_speed",
    "straightness",
    "mean_turn_angle",
    "turn_bias",
    "persistence_1s",
    "reorient_count",
    "mean_alignment",
    "near_wall_frac",
    "wall_contact_frac",
]

PERSISTENCE_LAG = 100        # steps -> 100 * 0.01 s = 1 second
WALL_TOUCH_DIST = 0.5        # um, "touching the wall" distance
REORIENT_ANGLE = np.pi / 3   # a turn bigger than 60 degrees in one step = reorientation


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def extract_features(xs: np.ndarray, ys: np.ndarray, W: float, dt: float = DT) -> Dict[str, float]:
    """Turn one trajectory (positions over time) into a dict of 11 features."""
    dx, dy = np.diff(xs), np.diff(ys)
    step = np.hypot(dx, dy)
    speeds = step / dt

    # --- Speed ---
    mean_speed = float(speeds.mean())
    std_speed = float(speeds.std())
    cv_speed = std_speed / mean_speed if mean_speed > 1e-9 else 0.0

    # --- Path shape (computed from the direction of motion) ---
    heading = np.arctan2(dy, dx)                  # direction of each step
    turn = _wrap_angle(np.diff(heading))          # change of direction between steps

    path_length = step.sum()
    net_disp = np.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
    straightness = float(net_disp / path_length) if path_length > 0 else 0.0

    mean_turn_angle = float(np.abs(turn).mean())
    turn_bias = float(turn.mean())                # >0 = tends to turn left, <0 = right

    lag = PERSISTENCE_LAG
    persistence_1s = float(np.mean(np.cos(heading[lag:] - heading[:-lag])))

    # Count big direction changes (tumbles), ignoring the instant a cell bounces off a wall
    y_mid = ys[1:-1]
    away_from_wall = (y_mid > WALL_TOUCH_DIST) & (y_mid < W - WALL_TOUCH_DIST)
    reorient_count = int(np.sum((np.abs(turn) > REORIENT_ANGLE) & away_from_wall))

    # --- Wall contact ---
    mean_alignment = float(np.abs(np.cos(heading)).mean())   # 1 = swimming parallel to the walls
    y_after = ys[1:]
    near_wall_frac = float(np.mean((y_after < 0.15 * W) | (y_after > 0.85 * W)))
    wall_contact_frac = float(np.mean((y_after < WALL_TOUCH_DIST) | (y_after > W - WALL_TOUCH_DIST)))

    return {
        "mean_speed": mean_speed,
        "std_speed": std_speed,
        "cv_speed": cv_speed,
        "straightness": straightness,
        "mean_turn_angle": mean_turn_angle,
        "turn_bias": turn_bias,
        "persistence_1s": persistence_1s,
        "reorient_count": reorient_count,
        "mean_alignment": mean_alignment,
        "near_wall_frac": near_wall_frac,
        "wall_contact_frac": wall_contact_frac,
    }
