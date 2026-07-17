"""
03_compare_classifiers.py

Compares Random Forest, XGBoost, and SVM using StratifiedGroupKFold grouped
by source file — NOT the plain StratifiedKFold used in the original repo
script, which let windows from the same recording appear in both train and
test within a fold. That version reproduces the exact leakage pattern
flagged earlier (99.9-100% accuracy that didn't hold up).

Saves classifier_comparison.csv and classifier_comparison.png, plus the
out-of-fold predictions needed for an honest confusion matrix in script 04.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

from config import OUT_DIR, RANDOM_STATE

DROP_COLS = [
    "session", "file_name", "relative_path", "point_id", "x", "y",
    "group_id", "window_start", "window_end", "grid_label",
]


def build_preprocess(num_cols, cat_cols):
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


def main():
    df = pd.read_csv(OUT_DIR / "feature_table.csv")
    df["grid_label"] = df["place"].astype(str) + "_P" + df["point_id"].astype(int).astype(str).str.zfill(2)

    X = df.drop(columns=DROP_COLS, errors="ignore")
    groups = df["group_id"]
    y_str = df["grid_label"].to_numpy()

    # Global encoder: only used for saving artifacts for script 04
    # (confusion matrix), not for training.
    le = LabelEncoder()
    le.fit(y_str)

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = ["place"]
    preprocess = build_preprocess(num_cols, cat_cols)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced"),
        "XGBoost": XGBClassifier(n_estimators=300, random_state=RANDOM_STATE, eval_metric="mlogloss"),
        "SVM (RBF)": SVC(kernel="rbf", C=10, random_state=RANDOM_STATE),
    }

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    oof_preds_str = {}

    for name, base_model in models.items():
        preds_full = np.full(len(y_str), fill_value="", dtype=object)
        fold_acc, fold_f1 = [], []

        for train_idx, test_idx in cv.split(X, y_str, groups):
            y_train_str = y_str[train_idx]
            y_test_str = y_str[test_idx]

            # Fold-local label encoding, fit only on this fold's training
            # labels, so it's always a dense 0..k-1 range by construction.
            # A *global* encoding can't guarantee that: several grid points
            # here (Scotiabank/Toronto) have only one file/group each, so
            # whichever fold holds that file out has zero training examples
            # of that class, leaving a gap in the class codes. XGBoost
            # requires contiguous 0..n-1 labels on every fit() call and
            # raises "Invalid classes inferred..." on that gap - Random
            # Forest and SVM tolerate it silently, which is why only
            # XGBoost crashed. Predictions are mapped back to label
            # strings immediately after predicting, so scoring is unaffected.
            fold_le = LabelEncoder().fit(y_train_str)
            y_train_local = fold_le.transform(y_train_str)

            pipe = Pipeline([("preprocess", preprocess), ("model", clone(base_model))])
            pipe.fit(X.iloc[train_idx], y_train_local)
            pred_local = pipe.predict(X.iloc[test_idx])
            pred_str = fold_le.inverse_transform(pred_local)

            preds_full[test_idx] = pred_str
            fold_acc.append(accuracy_score(y_test_str, pred_str))
            fold_f1.append(f1_score(y_test_str, pred_str, average="macro"))

        oof_preds_str[name] = preds_full
        results.append({
            "Model": name,
            "Mean Acc": np.mean(fold_acc),
            "Std Acc": np.std(fold_acc),
            "Mean Macro-F1": np.mean(fold_f1),
        })
        print(f"{name}: Acc={np.mean(fold_acc):.3f} +/- {np.std(fold_acc):.3f}  Macro-F1={np.mean(fold_f1):.3f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "classifier_comparison.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.bar(results_df["Model"], results_df["Mean Acc"], yerr=results_df["Std Acc"], capsize=5)
    plt.ylabel("Accuracy (grouped CV)")
    plt.title("Classifier Comparison - Grouped CV (leakage-safe)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "classifier_comparison.png", dpi=150)
    print(f"\nSaved: {OUT_DIR / 'classifier_comparison.csv'} and classifier_comparison.png")

    # Save out-of-fold predictions (global-encoded) + label encoder for
    # script 04's confusion matrix. Safe to transform here: every predicted
    # string is necessarily one of the labels the global encoder already knows.
    oof_preds_encoded = {name: le.transform(preds) for name, preds in oof_preds_str.items()}
    y_true_encoded = le.transform(y_str)
    joblib.dump({"y_true": y_true_encoded, "oof_preds": oof_preds_encoded}, OUT_DIR / "oof_predictions.pkl")
    joblib.dump(le, OUT_DIR / "label_encoder.pkl")


if __name__ == "__main__":
    main()
