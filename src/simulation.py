"""
BacMotionAI — Physics-Based Bacterial Simulation Engine

Simulates E. coli run-and-tumble motion in confined micro-channels
with hydrodynamic wall interactions and chemical modulation.

The simulator is only used to CREATE the dataset. The machine-learning task is:
    given one trajectory, predict which chemical condition it came from.
"""
import numpy as np
from typing import Tuple, Optional

# Physics constants
V0 = 20.0          # Intrinsic swimming speed (um/s)
DR = 0.2           # Rotational diffusion coefficient (rad^2/s)
DT = 0.01          # Time step (s)
T_TOTAL = 6.0      # Total simulation time (s)
STEPS = int(T_TOTAL / DT)
L = 200.0          # Channel length (um)
A = 1.0            # Cell body radius (um)
KAPPA = 6.0        # Hydrodynamic torque strength
EPS = 0.12         # Near-wall speed enhancement

# Chemical parameter distributions (mean, std)
CHEM_STATS = {
    'none':       {'v_mu': 1.00, 'v_sig': 0.05, 'Dr_mu': 1.0,  'Dr_sig': 0.1,
                   'drift_mu':  0.00, 'drift_sig': 0.05},
    'attractant': {'v_mu': 1.12, 'v_sig': 0.08, 'Dr_mu': 0.65, 'Dr_sig': 0.12,
                   'drift_mu':  0.28, 'drift_sig': 0.08},
    'repellent':  {'v_mu': 0.90, 'v_sig': 0.07, 'Dr_mu': 1.45, 'Dr_sig': 0.15,
                   'drift_mu': -0.22, 'drift_sig': 0.07},
    'antibiotic': {'v_mu': 0.52, 'v_sig': 0.10, 'Dr_mu': 2.40, 'Dr_sig': 0.30,
                   'drift_mu':  0.00, 'drift_sig': 0.04},
}

TUMBLE_RATES = {
    'none': 0.005,
    'attractant': 0.003,
    'repellent': 0.005,
    'antibiotic': 0.010,
}

CHEMICAL_LABELS = ['none', 'attractant', 'repellent', 'antibiotic']


def simulate_bacterium(
    W: float,
    hydro: bool = False,
    chemical: str = 'none',
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a single bacterium trajectory in a confined channel.

    Returns unwrapped x positions (no periodic jumps) for correct
    speed computation, plus y positions and orientation angles.
    """
    rng = np.random.RandomState(seed)

    cs = CHEM_STATS[chemical]
    v_scale = np.clip(rng.normal(cs['v_mu'], cs['v_sig']), 0.3, 1.5)
    Dr_eff = np.clip(rng.normal(cs['Dr_mu'], cs['Dr_sig']), 0.1, 5.0) * DR
    drift = rng.normal(cs['drift_mu'], cs['drift_sig'])
    p_tumble = TUMBLE_RATES[chemical]

    # Initial state
    x = rng.rand() * L
    y = rng.rand() * W
    theta = rng.rand() * 2 * np.pi

    xs, ys, thetas = [x], [y], [theta]

    for _ in range(STEPS):
        # Tumbling event
        if rng.rand() < p_tumble:
            theta = rng.rand() * 2 * np.pi

        # Hydrodynamic wall interaction
        omega = 0.0
        v_eff = V0 * v_scale

        if hydro:
            h = max(min(y, W - y), A)
            omega = -KAPPA * (A / h) ** 2 * np.sin(2 * theta)
            v_eff = V0 * v_scale * (1 + EPS * (A / h))

        # Update orientation
        theta += omega * DT + drift * DT + np.sqrt(2 * Dr_eff * DT) * rng.randn()

        # Update position
        dx = v_eff * np.cos(theta) * DT
        dy = v_eff * np.sin(theta) * DT

        # x: store UNWRAPPED position for correct speed calculation
        # (we track displacement, not wrapped coordinate)
        x += dx
        y += dy

        # y: reflecting walls
        if y < 0:
            y = -y
            theta = -theta
        elif y > W:
            y = 2 * W - y
            theta = -theta

        xs.append(x)
        ys.append(y)
        thetas.append(theta)

    return np.array(xs), np.array(ys), np.array(thetas)
