"""
Step 3 - Train and tune models.

- Stratified 80/20 train/test split (the test set is NOT touched here).
- Each model is a Pipeline(StandardScaler -> classifier).
- GridSearchCV with 5-fold stratified CV on the TRAINING set picks hyper-parameters.
- The best model is chosen by CROSS-VALIDATION score (not by test score).

Run: python scripts/train.py
"""
import os
import sys
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.data import load_dataset, split_data, MODELS_DIR, RESULTS_DIR
from src.models import get_models_and_grids, model_filename


def main():
    df = load_dataset()
    X_train, X_test, y_train, y_test, _ = split_data(df)
    print(f"Train: {len(X_train)} rows | Test (held out): {len(X_test)} rows")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []
    for name, (pipe, grid) in get_models_and_grids().items():
        print(f"\nTuning {name} ...")
        search = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=1)
        search.fit(X_train, y_train)

        best_idx = search.best_index_
        cv_mean = search.cv_results_["mean_test_score"][best_idx]
        cv_std = search.cv_results_["std_test_score"][best_idx]
        params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        print(f"  best params : {params}")
        print(f"  CV macro-F1 : {cv_mean:.3f} +/- {cv_std:.3f}")

        joblib.dump(search.best_estimator_, os.path.join(MODELS_DIR, model_filename(name)))
        rows.append({"model": name, "cv_macro_f1": round(cv_mean, 4),
                     "cv_std": round(cv_std, 4), "best_params": json.dumps(params, default=str)})

    res = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False)
    res.to_csv(os.path.join(RESULTS_DIR, "cv_results.csv"), index=False)

    best_name = res.iloc[0]["model"]
    best_model = joblib.load(os.path.join(MODELS_DIR, model_filename(best_name)))
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
    with open(os.path.join(RESULTS_DIR, "best_model.json"), "w") as f:
        json.dump({"best_model": best_name, "selected_by": "5-fold CV macro-F1 on training set"}, f, indent=2)

    print("\n" + "=" * 60)
    print(res[["model", "cv_macro_f1", "cv_std"]].to_string(index=False))
    print(f"\nBest model (by CV): {best_name}")
    print("Next: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
