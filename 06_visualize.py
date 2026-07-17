"""
06_visualize.py

Generates the descriptive/diagnostic plots: feature correlation heatmap,
Bnorm distribution by place, grid position overlay, predicted-vs-actual
coordinates, and spatial error map. Run after 01 and 05.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import OUT_DIR


def main():
    ml_df = pd.read_csv(OUT_DIR / "feature_table.csv")

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = ml_df.select_dtypes(include=np.number).drop(
        columns=["point_id", "x", "y", "window_start", "window_end"], errors="ignore"
    ).corr()
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot_correlation.png", dpi=150)
    plt.close()

    # Boxplot of Bnorm_mean by place
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=ml_df, x="place", y="Bnorm_mean")
    plt.xticks(rotation=30, ha="right")
    plt.title("Bnorm mean distribution by place")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot_boxplot.png", dpi=150)
    plt.close()

    # Position overlay
    plt.figure(figsize=(7, 6))
    for place, g in ml_df.groupby("place"):
        plt.scatter(g["x"], g["y"], label=place, alpha=0.6)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Grid Position Overlay by Place")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot_position_overlay.png", dpi=150)
    plt.close()

    print(f"Saved plot_correlation.png, plot_boxplot.png, plot_position_overlay.png in {OUT_DIR}")

    # Predicted vs actual + spatial error (needs regression predictions from script 05)
    pred_path = OUT_DIR / "regression_predictions.csv"
    if not pred_path.exists():
        print(f"\n{pred_path} not found - run 05_train_regression.py first to get these two plots.")
        return

    compare_df = pd.read_csv(pred_path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].scatter(compare_df["x"], compare_df["x_pred"], alpha=0.5)
    axes[0].plot([compare_df["x"].min(), compare_df["x"].max()],
                 [compare_df["x"].min(), compare_df["x"].max()], "r--")
    axes[0].set_xlabel("Actual x")
    axes[0].set_ylabel("Predicted x")
    axes[0].set_title("x: predicted vs actual")

    axes[1].scatter(compare_df["y"], compare_df["y_pred"], alpha=0.5)
    axes[1].plot([compare_df["y"].min(), compare_df["y"].max()],
                 [compare_df["y"].min(), compare_df["y"].max()], "r--")
    axes[1].set_xlabel("Actual y")
    axes[1].set_ylabel("Predicted y")
    axes[1].set_title("y: predicted vs actual")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot_pred_vs_actual.png", dpi=150)
    plt.close()

    compare_df["error"] = np.sqrt(
        (compare_df["x"] - compare_df["x_pred"]) ** 2 + (compare_df["y"] - compare_df["y_pred"]) ** 2
    )

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(compare_df["x"], compare_df["y"], c=compare_df["error"], cmap="viridis", s=60)
    plt.colorbar(sc, label="Euclidean error (grid units)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Spatial Error - Actual Grid Positions")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot_spatial_error.png", dpi=150)
    plt.close()

    print(f"Saved plot_pred_vs_actual.png, plot_spatial_error.png in {OUT_DIR}")


if __name__ == "__main__":
    main()
