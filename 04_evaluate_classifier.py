"""
04_evaluate_classifier.py

Two diagnostics for the Random Forest:

  - Feature importance: fit on all data, purely for interpretability
    (which features the model leans on). This is descriptive, not an
    evaluation number.
  - Confusion matrix: built from the grouped out-of-fold predictions saved
    by 03_compare_classifiers.py, so it reflects generalization performance
    rather than the model re-predicting data it memorized. The original
    repo script computed this from full-data predictions of the fitted
    model, which is a memorization baseline, not a test result.
"""

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import OUT_DIR


def main():
    rf_pipe = joblib.load(OUT_DIR / "rf_model.pkl")
    oof = joblib.load(OUT_DIR / "oof_predictions.pkl")
    le = joblib.load(OUT_DIR / "label_encoder.pkl")

    # --- Feature importance -------------------------------------------------
    preprocess = rf_pipe.named_steps["preprocess"]
    model = rf_pipe.named_steps["model"]

    num_cols = preprocess.transformers_[0][2]
    cat_cols = preprocess.transformers_[1][2]
    cat_names = list(preprocess.named_transformers_["cat"].get_feature_names_out(cat_cols))
    feature_names = list(num_cols) + cat_names

    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(12, 5))
    importances.head(25).plot(kind="bar")
    plt.title("Feature Importance - Random Forest (top 25)")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot_feature_importance.png", dpi=150)
    print(f"Saved: {OUT_DIR / 'plot_feature_importance.png'}")

    # --- Confusion matrix (grouped out-of-fold predictions, Random Forest) --
    y_true = oof["y_true"]
    rf_oof = oof["oof_preds"]["Random Forest"]

    cm = confusion_matrix(y_true, rf_oof)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title("Confusion Matrix - Random Forest (grouped out-of-fold predictions)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot_confusion_rf.png", dpi=150)
    print(f"Saved: {OUT_DIR / 'plot_confusion_rf.png'}")


if __name__ == "__main__":
    main()
