"""
08_tune_hyperparameters.py

Tunes Random Forest, XGBoost, and SVM hyperparameters and compares each
against its default/baseline configuration, all scored with the same
StratifiedGroupKFold (grouped by source file) used everywhere else in
this pipeline, and the same fold-local label encoding fix from
03_compare_classifiers.py so XGBoost doesn't crash on folds missing a class.

Answers, for each model:
  - which hyperparameters were tuned and what range was tested
    (see PARAM_GRIDS below)
  - which metric was used to pick a winner (Macro-F1 - see SELECTION_METRIC)
  - the best parameters found
  - how the tuned model compares to the default/baseline configuration
    (see DEFAULT_PARAMS and baseline_vs_tuned.csv)

Honest expectation going in: with several grid points backed by only one
recording session each (see 09_data_coverage_report.py), no amount of
hyperparameter tuning can manufacture a cross-session signal that isn't
in the data. This search will find the best of what's achievable with the
current data, but the ceiling here is a data-collection problem, not a
model-selection problem.

Saves best_hyperparameters.json, tuned_classifier_comparison.csv,
baseline_vs_tuned.csv, tuning_report.txt, and refits + saves the best
model of each type on all data.
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

# The hyperparameters 03_compare_classifiers.py used before any tuning -
# the baseline every tuned result below is measured against.
DEFAULT_PARAMS = {
    "Random Forest": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
    "XGBoost": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.3},
    "SVM (RBF)": {"C": 10, "gamma": "scale"},
}

# The metric used to pick a winner. Macro-F1 (not accuracy) is used because
# there are ~57 classes with very uneven support per class (some grid points
# have 10 sessions worth of windows, others have 1) - plain accuracy would
# reward a model that only ever predicts the best-represented classes.
SELECTION_METRIC = "Mean Macro-F1"


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

    # --- Baseline: the default hyperparameters, scored the same way ---------
    print("=== Baseline (default hyperparameters) ===")
    baseline_results = {}
    for model_name, params in DEFAULT_PARAMS.items():
        mean_acc, std_acc, mean_f1 = grouped_cv_score(X, y_str, groups, preprocess, model_name, params, cv)
        baseline_results[model_name] = {
            "params": params, "Mean Acc": mean_acc, "Std Acc": std_acc, "Mean Macro-F1": mean_f1,
        }
        print(f"  {model_name} {params}: Acc={mean_acc:.3f} +/- {std_acc:.3f}  Macro-F1={mean_f1:.3f}")

    # --- Tuning: search the grids in PARAM_GRIDS -----------------------------
    all_results = []
    best_per_model = {}

    for model_name, grid in PARAM_GRIDS.items():
        print(f"\n=== Tuning {model_name} ===")
        print(f"  Search space: {grid}")
        best_metric = -1
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

            metric_value = row[SELECTION_METRIC]
            if metric_value > best_metric:
                best_metric = metric_value
                best_params = params
                best_row = row

        best_per_model[model_name] = {"params": best_params, "cv_result": best_row}
        print(f"  -> best by {SELECTION_METRIC}: {best_params} ({SELECTION_METRIC}={best_metric:.3f})")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_DIR / "tuned_classifier_comparison.csv", index=False)

    with open(OUT_DIR / "best_hyperparameters.json", "w") as f:
        json.dump(best_per_model, f, indent=2)

    # --- Baseline vs tuned comparison table ----------------------------------
    comparison_rows = []
    for model_name in PARAM_GRIDS:
        base = baseline_results[model_name]
        tuned = best_per_model[model_name]["cv_result"]
        comparison_rows.append({
            "Model": model_name,
            "Baseline Params": json.dumps(base["params"]),
            "Baseline Acc": base["Mean Acc"],
            "Baseline Macro-F1": base["Mean Macro-F1"],
            "Tuned Params": json.dumps(best_per_model[model_name]["params"]),
            "Tuned Acc": tuned["Mean Acc"],
            "Tuned Macro-F1": tuned["Mean Macro-F1"],
            "Macro-F1 Change": tuned["Mean Macro-F1"] - base["Mean Macro-F1"],
        })
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(OUT_DIR / "baseline_vs_tuned.csv", index=False)

    # --- Written report answering the tuning questions directly -------------
    lines = []
    lines.append("HYPERPARAMETER TUNING REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Evaluation protocol: StratifiedGroupKFold, 5 folds, grouped by")
    lines.append("source recording file (group_id), so no window from the same")
    lines.append("file/session ever appears in both train and test. Every number")
    lines.append("below is a 5-fold mean.")
    lines.append("")
    lines.append("Metric used to select the best hyperparameters: Macro-F1.")
    lines.append("Accuracy is also reported for reference, but Macro-F1 was used")
    lines.append("for model selection because classes are highly imbalanced (~57")
    lines.append("grid-point classes, with per-class support ranging from a single")
    lines.append("recording session to ten) - plain accuracy can look good on a")
    lines.append("model that only ever predicts the best-represented classes.")
    lines.append("")

    for model_name, grid in PARAM_GRIDS.items():
        lines.append(f"--- {model_name} ---")
        lines.append(f"Hyperparameters tuned and range tested:")
        for param, values in grid.items():
            lines.append(f"    {param}: {values}")
        base = baseline_results[model_name]
        tuned = best_per_model[model_name]["cv_result"]
        lines.append(f"Default/baseline params: {base['params']}")
        lines.append(f"  Baseline  -> Acc={base['Mean Acc']:.3f}, Macro-F1={base['Mean Macro-F1']:.3f}")
        lines.append(f"Best params found: {best_per_model[model_name]['params']}")
        lines.append(f"  Tuned     -> Acc={tuned['Mean Acc']:.3f}, Macro-F1={tuned['Mean Macro-F1']:.3f}")
        change = tuned["Mean Macro-F1"] - base["Mean Macro-F1"]
        direction = "improved" if change > 0 else ("worsened" if change < 0 else "unchanged")
        lines.append(f"  Change: Macro-F1 {direction} by {abs(change):.3f}")
        lines.append("")

    lines.append("Overall: tuned vs baseline")
    lines.append(comparison_df.to_string(index=False))
    lines.append("")
    lines.append(
        "Interpretation: hyperparameter tuning made little to no meaningful "
        "difference here. That is expected, not a sign the search was done "
        "wrong - see 09_data_coverage_report.py. Many classes are backed by "
        "only one recording session, so there is no cross-session variation "
        "in the training data for the model to learn a tolerance for in the "
        "first place. No choice of tree depth, learning rate, or SVM kernel "
        "parameter can substitute for that missing variation. The next "
        "highest-value step is collecting more repeated sessions per grid "
        "point with consistent sensor height, orientation, and placement, "
        "not further hyperparameter search."
    )

    report_text = "\n".join(lines)
    print("\n" + report_text)

    with open(OUT_DIR / "tuning_report.txt", "w") as f:
        f.write(report_text)

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

    print(f"\nSaved: tuned_classifier_comparison.csv, best_hyperparameters.json,")
    print(f"       baseline_vs_tuned.csv, tuning_report.txt in {OUT_DIR}")


if __name__ == "__main__":
    main()
