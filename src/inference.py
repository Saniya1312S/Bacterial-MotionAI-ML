"""Use a trained model on a new trajectory."""
import os
import joblib
import numpy as np
import pandas as pd

from .data import MODELS_DIR
from .features import extract_features, FEATURE_COLS


def load_best_model(models_dir: str = MODELS_DIR):
    path = os.path.join(models_dir, "best_model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError("best_model.joblib not found. Run: python scripts/train.py")
    return joblib.load(path)


def predict_trajectory(model, xs: np.ndarray, ys: np.ndarray, W: float):
    """Return (predicted_class, {class: probability}, feature_dict) for one trajectory."""
    feats = extract_features(xs, ys, W)
    X = pd.DataFrame([feats])[FEATURE_COLS]
    proba = model.predict_proba(X)[0]
    probs = dict(zip(model.classes_, proba.round(4)))
    return model.predict(X)[0], probs, feats
