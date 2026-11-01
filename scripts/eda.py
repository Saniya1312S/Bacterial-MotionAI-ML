"""
Step 2 - Exploratory data analysis.

Saves: images/eda_feature_distributions.png, images/eda_correlation.png
Run: python scripts/eda.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import load_dataset, IMAGES_DIR, TARGET
from src.features import FEATURE_COLS
from src.models import CLASS_NAMES


def main():
    df = load_dataset()
    os.makedirs(IMAGES_DIR, exist_ok=True)
    print(f"Rows: {len(df)} | Features: {len(FEATURE_COLS)}")
    print(f"Missing values: {int(df[FEATURE_COLS].isna().sum().sum())}")
    print("\nMean of each feature per class:")
    print(df.groupby(TARGET)[FEATURE_COLS].mean().round(3).T.to_string())

    # 1. Distribution of every feature by class
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for ax, col in zip(axes.ravel(), FEATURE_COLS):
        sns.boxplot(data=df, x=TARGET, y=col, hue=TARGET, order=CLASS_NAMES, hue_order=CLASS_NAMES,
                    ax=ax, palette="Set2", fliersize=2, legend=False)
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
    for ax in axes.ravel()[len(FEATURE_COLS):]:
        ax.axis("off")
    plt.suptitle("Feature distributions by chemical condition", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "eda_feature_distributions.png"), dpi=130)
    plt.close()

    # 2. Correlation heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df[FEATURE_COLS].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature correlation", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "eda_correlation.png"), dpi=130)
    plt.close()

    corr = df[FEATURE_COLS].corr().abs()
    pairs = [(a, b, float(round(corr.loc[a, b], 2))) for i, a in enumerate(FEATURE_COLS)
             for b in FEATURE_COLS[i + 1:] if corr.loc[a, b] > 0.8]
    print("\nHighly correlated pairs (|r| > 0.8):", pairs if pairs else "none")
    print("Saved EDA plots to images/")


if __name__ == "__main__":
    main()
