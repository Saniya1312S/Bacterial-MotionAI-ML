"""
Step 4 - Evaluate on the held-out TEST set (used once, after all tuning is finished).

Produces:
  results/test_results.csv     test metrics + 95% bootstrap confidence intervals
  results/baselines.csv        simple baselines for comparison
  results/error_analysis.csv   accuracy by channel width and wall setting
  results/results.md           tables ready to paste into the README
  images/*.png                 confusion matrix, ROC, model comparison, feature importance, error analysis

Run: python scripts/evaluate.py
"""
import os
import sys
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, roc_curve, auc,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)

from src.data import load_dataset, split_data, MODELS_DIR, RESULTS_DIR, IMAGES_DIR
from src.features import FEATURE_COLS
from src.models import get_models_and_grids, model_filename, CLASS_NAMES

SPEED_ONLY = ["mean_speed", "std_speed", "cv_speed"]


def bootstrap_ci(y_true, y_pred, metric, n=1000, seed=42):
    """95% confidence interval by resampling the test predictions."""
    rng = np.random.RandomState(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    vals = []
    for _ in range(n):
        idx = rng.randint(0, len(y_true), len(y_true))
        vals.append(metric(y_true[idx], y_pred[idx]))
    return np.percentile(vals, [2.5, 97.5])


def macro_f1(a, b):
    return f1_score(a, b, average="macro")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    df = load_dataset()
    X_train, X_test, y_train, y_test, df_test = split_data(df)
    best_name = json.load(open(os.path.join(RESULTS_DIR, "best_model.json")))["best_model"]
    print(f"Test set: {len(X_test)} rows | Best model chosen by CV: {best_name}\n")

    # ---------------- Baselines ----------------
    baselines = {
        "Majority class": (DummyClassifier(strategy="most_frequent"), FEATURE_COLS),
        "Decision tree (depth 3), all features": (DecisionTreeClassifier(max_depth=3, random_state=42), FEATURE_COLS),
        "Decision tree (depth 3), mean_speed only": (DecisionTreeClassifier(max_depth=3, random_state=42), ["mean_speed"]),
        "Logistic Regression, speed features only (3)": (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), SPEED_ONLY),
    }
    base_rows = []
    for name, (clf, cols) in baselines.items():
        clf.fit(X_train[cols], y_train)
        pred = clf.predict(X_test[cols])
        base_rows.append({"baseline": name, "accuracy": accuracy_score(y_test, pred),
                          "macro_f1": macro_f1(y_test, pred)})
    base_df = pd.DataFrame(base_rows)
    base_df.to_csv(os.path.join(RESULTS_DIR, "baselines.csv"), index=False)
    print("BASELINES\n" + base_df.round(3).to_string(index=False) + "\n")

    # ---------------- Tuned models on the test set ----------------
    cv_df = pd.read_csv(os.path.join(RESULTS_DIR, "cv_results.csv")).set_index("model")
    rows, preds, probas = [], {}, {}
    for name in get_models_and_grids():
        model = joblib.load(os.path.join(MODELS_DIR, model_filename(name)))
        pred, proba = model.predict(X_test), model.predict_proba(X_test)
        preds[name], probas[name] = pred, proba
        acc_lo, acc_hi = bootstrap_ci(y_test, pred, accuracy_score)
        f1_lo, f1_hi = bootstrap_ci(y_test, pred, macro_f1)
        rows.append({
            "model": name,
            "train_acc": accuracy_score(y_train, model.predict(X_train)),
            "cv_macro_f1": cv_df.loc[name, "cv_macro_f1"],
            "test_acc": accuracy_score(y_test, pred),
            "test_acc_ci_low": acc_lo, "test_acc_ci_high": acc_hi,
            "test_macro_f1": macro_f1(y_test, pred),
            "test_f1_ci_low": f1_lo, "test_f1_ci_high": f1_hi,
            "roc_auc_macro": roc_auc_score(y_test, proba, multi_class="ovr", labels=model.classes_),
        })
    res = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False)
    res["train_test_gap"] = res["train_acc"] - res["test_acc"]
    res.to_csv(os.path.join(RESULTS_DIR, "test_results.csv"), index=False)
    print("TEST RESULTS\n" + res.round(3).to_string(index=False) + "\n")

    best_model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    best_pred = preds[best_name]
    print(f"Per-class report ({best_name}):")
    print(classification_report(y_test, best_pred, labels=CLASS_NAMES, digits=3))

    # ---------------- Plots ----------------
    # 1. Confusion matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix(y_test, best_pred, labels=CLASS_NAMES),
                           display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion matrix - {best_name} (test set)", fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(IMAGES_DIR, "confusion_matrix.png"), dpi=130); plt.close()

    # 2. Model comparison with confidence intervals
    fig, ax = plt.subplots(figsize=(9, 5))
    r = res.sort_values("test_macro_f1")
    err = [r["test_macro_f1"] - r["test_f1_ci_low"], r["test_f1_ci_high"] - r["test_macro_f1"]]
    ax.barh(r["model"], r["test_macro_f1"], xerr=err, color="#3B82F6", alpha=0.85, capsize=4)
    ax.axvline(base_df.loc[base_df.baseline.str.contains("speed features only"), "macro_f1"].iloc[0],
               color="red", ls="--", label="speed-only baseline")
    ax.set_xlim(0.5, 1.0); ax.set_xlabel("Test macro-F1 (95% bootstrap CI)")
    ax.set_title("Model comparison", fontweight="bold"); ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(os.path.join(IMAGES_DIR, "model_comparison.png"), dpi=130); plt.close()

    # 3. ROC curves, one-vs-rest
    fig, ax = plt.subplots(figsize=(7, 6))
    classes = list(best_model.classes_)
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve((y_test == cls).astype(int), probas[best_name][:, i])
        ax.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curves (one-vs-rest) - {best_name}", fontweight="bold"); ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(os.path.join(IMAGES_DIR, "roc_curves.png"), dpi=130); plt.close()

    # 4. Permutation importance (more trustworthy than Gini importance)
    perm = permutation_importance(best_model, X_test, y_test, scoring="f1_macro",
                                  n_repeats=10, random_state=42)
    imp = pd.Series(perm.importances_mean, index=FEATURE_COLS).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp.index, imp.values, xerr=perm.importances_std[[FEATURE_COLS.index(c) for c in imp.index]],
            color="#06B6D4")
    ax.set_xlabel("Drop in test macro-F1 when the feature is shuffled")
    ax.set_title(f"Permutation importance - {best_name}", fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(IMAGES_DIR, "feature_importance.png"), dpi=130); plt.close()
    print("Permutation importance (top 5):\n" + imp.sort_values(ascending=False).head(5).round(3).to_string() + "\n")

    # 5. Error analysis by channel width and wall setting
    ea = df_test.assign(correct=(best_pred == y_test.values))
    by_width = ea.groupby("channel_width")["correct"].mean().rename("accuracy").reset_index()
    by_width["group"] = "width " + by_width["channel_width"].astype(str) + " um"
    by_hydro = ea.groupby("hydro")["correct"].mean().rename("accuracy").reset_index()
    by_hydro["group"] = np.where(by_hydro["hydro"], "wall interaction ON", "wall interaction OFF")
    err_df = pd.concat([by_width[["group", "accuracy"]], by_hydro[["group", "accuracy"]]])
    err_df.to_csv(os.path.join(RESULTS_DIR, "error_analysis.csv"), index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(err_df["group"], err_df["accuracy"], color=["#8B5CF6"] * len(by_width) + ["#F59E0B"] * 2)
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("Test accuracy")
    ax.set_title(f"Where does {best_name} make mistakes?", fontweight="bold")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "error_analysis.png"), dpi=130); plt.close()
    print("Accuracy by group:\n" + err_df.round(3).to_string(index=False) + "\n")

    # ---------------- Markdown tables for the README ----------------
    pc = classification_report(y_test, best_pred, labels=CLASS_NAMES, output_dict=True)
    with open(os.path.join(RESULTS_DIR, "results.md"), "w", encoding="utf-8") as f:
        f.write("### Baselines (test set)\n\n| Baseline | Accuracy | Macro-F1 |\n|:--|:-:|:-:|\n")
        for _, b in base_df.iterrows():
            f.write(f"| {b.baseline} | {b.accuracy:.1%} | {b.macro_f1:.3f} |\n")
        f.write("\n### Tuned models\n\n| Model | CV macro-F1 | Train acc | Test acc (95% CI) | Test macro-F1 | ROC-AUC |\n|:--|:-:|:-:|:-:|:-:|:-:|\n")
        for _, m in res.iterrows():
            f.write(f"| {m.model} | {m.cv_macro_f1:.3f} | {m.train_acc:.1%} | {m.test_acc:.1%} "
                    f"({m.test_acc_ci_low:.1%} - {m.test_acc_ci_high:.1%}) | {m.test_macro_f1:.3f} | {m.roc_auc_macro:.3f} |\n")
        f.write(f"\n### Per-class results ({best_name})\n\n| Class | Precision | Recall | F1 |\n|:--|:-:|:-:|:-:|\n")
        for c in CLASS_NAMES:
            f.write(f"| {c} | {pc[c]['precision']:.2f} | {pc[c]['recall']:.2f} | {pc[c]['f1-score']:.2f} |\n")
    print("Saved results/*.csv, results/results.md and images/*.png")


if __name__ == "__main__":
    main()
