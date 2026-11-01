# Models

`best_model.joblib` is the trained scikit-learn Pipeline (StandardScaler -> Random Forest).
The other models (`random_forest`, `gradient_boosting`, `logistic_regression`, `knn`) are created by
`python scripts/train.py` and are git-ignored to keep the repo small.

If you see a scikit-learn version warning when loading the model, just re-run `python scripts/train.py`.
