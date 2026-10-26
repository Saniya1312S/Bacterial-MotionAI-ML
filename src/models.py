"""Models and the small hyper-parameter grids that are searched with cross-validation.

Every model is a scikit-learn Pipeline (scaler -> classifier), so the scaler is
re-fitted inside every CV fold and can never see the validation data.
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

CLASS_NAMES = ["none", "attractant", "repellent", "antibiotic"]


def _pipe(clf) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def get_models_and_grids() -> dict:
    """Return {name: (pipeline, param_grid)}."""
    return {
        "Logistic Regression": (
            _pipe(LogisticRegression(max_iter=2000, random_state=42)),
            {"clf__C": [0.1, 1, 10]},
        ),
        "KNN": (
            _pipe(KNeighborsClassifier()),
            {"clf__n_neighbors": [5, 11, 21, 41]},
        ),
        "Random Forest": (
            _pipe(RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
            {"clf__max_depth": [4, 8, None], "clf__min_samples_leaf": [1, 5]},
        ),
        "Gradient Boosting": (
            _pipe(GradientBoostingClassifier(n_estimators=100, random_state=42)),
            {"clf__learning_rate": [0.05, 0.1], "clf__max_depth": [2, 3]},
        ),
    }


def model_filename(name: str) -> str:
    return name.lower().replace(" ", "_") + ".joblib"
