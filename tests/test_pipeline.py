"""A tiny end-to-end test: simulate -> features -> train -> predict."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from sklearn.model_selection import train_test_split
from src.simulation import simulate_bacterium, CHEMICAL_LABELS
from src.features import extract_features, FEATURE_COLS
from src.models import get_models_and_grids


def _tiny_dataset(n_per_class=25):
    rows = []
    for chem in CHEMICAL_LABELS:
        for s in range(n_per_class):
            xs, ys, _ = simulate_bacterium(W=30, chemical=chem, seed=1000 + s)
            row = extract_features(xs, ys, W=30)
            row["chemical"] = chem
            rows.append(row)
    return pd.DataFrame(rows)


def test_end_to_end_beats_random_guessing():
    df = _tiny_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[FEATURE_COLS], df["chemical"], test_size=0.3, random_state=0, stratify=df["chemical"])
    pipe, _ = get_models_and_grids()["Logistic Regression"]
    pipe.fit(X_tr, y_tr)
    assert pipe.score(X_te, y_te) > 0.5          # random guessing would be 0.25
    proba = pipe.predict_proba(X_te)
    assert proba.shape == (len(X_te), 4)
    assert abs(proba.sum(axis=1) - 1).max() < 1e-6
