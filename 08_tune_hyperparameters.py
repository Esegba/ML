"""
08_tune_hyperparameters.py

Small grid search over Random Forest, XGBoost, and SVM hyperparameters,
scored with the same StratifiedGroupKFold (grouped by source file) used
everywhere else in this pipeline, and the same fold-local label encoding
fix from 03_compare_classifiers.py so XGBoost doesn't crash on folds
missing a class.

Honest expectation going in: with several grid points backed by only one
recording session each (see 09_data_coverage_report.py), no amount of
hyperparameter tuning can manufacture a cross-session signal that isn't
in the data. This search will find the best of what's achievable with the
current data, but the ceiling here is a data-collection problem, not a
model-selection problem. Worth reading the coverage report alongside these
results.

Saves best_hyperparameters.json, tuned_classifier_comparison.csv, and
refits + saves the best model of each type on all data.
"""

import itertools
import json

import joblib
import numpy as np
import pandas as pd

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

# Kept intentionally small - this is a ~2600-row dataset with many classes
# backed by very few groups, so a large search mostly re-measures noise.
PARAM_GRIDS = {
    "Random Forest": {
        "n_estimators": [200, 400],
        "max_depth": [None, 15],
        "min_samples_leaf": [1, 3],
    },
    "XGBoost": {
        "n_estimators": [200, 400],
        "max_depth": [3, 6],
        "learning_rate": [0.05, 0.1],
    },
    "SVM (RBF)": {
        "C": [1, 10, 50],
        "gamma": ["scale", "auto"],
    },
}


def build_preprocess(num_cols, cat_cols):
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


def make_model(name, params):
    if name == "Random Forest":
        return RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", **params)
    if name == "XGBoost":
        return XGBClassifier(random_state=RANDOM_STATE, eval_metric="mlogloss", **params)
    if name == "SVM (RBF)":
        return SVC(kernel="rbf", random_state=RANDOM_STATE, **params)
    raise ValueError(name)


def param_combinations(grid):
    keys = list(grid.keys())
    for values in itertools.product(*grid.values()):
        yield dict(zip(keys, values))


def grouped_cv_score(X, y_str, groups, preprocess, model_name, params, cv):
    """Same grouped, fold-local-encoded CV loop as 03_compare_classifiers.py."""
    fold_acc, fold_f1 = [], []

    for train_idx, test_idx in cv.split(X, y_str, groups):
        y_train_str = y_str[train_idx]
        y_test_str = y_str[test_idx]

        fold_le = LabelEncoder().fit(y_train_str)
        y_train_local = fold_le.transform(y_train_str)

        model = make_model(model_name, params)
        pipe = Pipeline([("preprocess", clone(preprocess)), ("model", model)])
        pipe.fit(X.iloc[train_idx], y_train_local)

        pred_local = pipe.predict(X.iloc[test_idx])
        pred_str = fold_le.inverse_transform(pred_local)

        fold_acc.append(accuracy_score(y_test_str, pred_str))
        fold_f1.append(f1_score(y_test_str, pred_str, average="macro"))

    return np.mean(fold_acc), np.std(fold_acc), np.mean(fold_f1)


def main():
    df = pd.read_csv(OUT_DIR / "feature_table.csv")
    df["grid_label"] = df["place"].astype(str) + "_P" + df["point_id"].astype(int).astype(str).str.zfill(2)

    X = df.drop(columns=DROP_COLS, errors="ignore")
    y_str = df["grid_label"].to_numpy()
    groups = df["group_id"]

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = ["place"]
    preprocess = build_preprocess(num_cols, cat_cols)

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    all_results = []
    best_per_model = {}

    for model_name, grid in PARAM_GRIDS.items():
        print(f"\n=== Tuning {model_name} ===")
        best_f1 = -1
        best_params = None
        best_row = None

        for params in param_combinations(grid):
            mean_acc, std_acc, mean_f1 = grouped_cv_score(X, y_str, groups, preprocess, model_name, params, cv)
            row = {
                "Model": model_name, "Params": json.dumps(params),
                "Mean Acc": mean_acc, "Std Acc": std_acc, "Mean Macro-F1": mean_f1,
            }
            all_results.append(row)
            print(f"  {params}: Acc={mean_acc:.3f} +/- {std_acc:.3f}  Macro-F1={mean_f1:.3f}")

            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_params = params
                best_row = row

        best_per_model[model_name] = {"params": best_params, "cv_result": best_row}
        print(f"  -> best: {best_params} (Macro-F1={best_f1:.3f})")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_DIR / "tuned_classifier_comparison.csv", index=False)

    with open(OUT_DIR / "best_hyperparameters.json", "w") as f:
        json.dump(best_per_model, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'tuned_classifier_comparison.csv'} and best_hyperparameters.json")

    # Refit each model's best config on ALL data and save, for downstream reuse.
    # (Descriptive artifact, not a new evaluation number - the numbers above
    # are the ones to report.)
    le_global = LabelEncoder().fit(y_str)
    for model_name, info in best_per_model.items():
        model = make_model(model_name, info["params"])
        pipe = Pipeline([("preprocess", clone(preprocess)), ("model", model)])
        pipe.fit(X, le_global.transform(y_str))
        safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(pipe, OUT_DIR / f"tuned_{safe_name}_model.pkl")

    print("\nReminder: read 09_data_coverage_report.py's output alongside these numbers.")
    print("Several classes here have only one recording session, which caps how much")
    print("any amount of tuning can improve cross-session generalization.")


if __name__ == "__main__":
    main()
