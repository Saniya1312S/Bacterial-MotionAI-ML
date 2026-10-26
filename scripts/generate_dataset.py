"""
Step 1 - Generate the dataset.

Simulates 2,400 bacteria (6 channel widths x 4 chemicals x 2 wall settings x 50 cells),
turns each trajectory into 11 features and saves a CSV.

The TARGET is the chemical condition the simulator used (known ground truth).
channel_width / hydro / seed are saved as metadata only - they are NOT model inputs.

Run: python scripts/generate_dataset.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.simulation import simulate_bacterium, CHEMICAL_LABELS
from src.features import extract_features, FEATURE_COLS
from src.data import DATA_PATH

WIDTHS = [10, 20, 30, 50, 75, 100]   # channel widths in micrometres
HYDRO_OPTIONS = [False, True]        # hydrodynamic wall interaction off / on
SAMPLES_PER_COMBO = 50


def main():
    records, seed = [], 0
    total = len(WIDTHS) * len(CHEMICAL_LABELS) * len(HYDRO_OPTIONS) * SAMPLES_PER_COMBO
    print(f"Generating {total} trajectories...")

    for W in WIDTHS:
        for chem in CHEMICAL_LABELS:
            for hydro in HYDRO_OPTIONS:
                for _ in range(SAMPLES_PER_COMBO):
                    xs, ys, _ = simulate_bacterium(W, hydro, chem, seed=seed)
                    row = extract_features(xs, ys, W)
                    row.update(chemical=chem, channel_width=W, hydro=hydro, seed=seed)
                    records.append(row)
                    seed += 1
        print(f"  width {W:>3} um done ({len(records)}/{total})")

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    print(f"\nSaved {len(df)} rows x {len(FEATURE_COLS)} features -> {DATA_PATH}")
    print("\nClass balance:")
    print(df["chemical"].value_counts().to_string())
    print(f"\nMissing values: {int(df[FEATURE_COLS].isna().sum().sum())}")


if __name__ == "__main__":
    main()
