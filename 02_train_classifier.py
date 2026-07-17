"""
02_train_classifier.py

Trains a Random Forest to predict grid point ("place_Pxx"), and reports
two numbers side by side:

  - Sanity-check accuracy (plain random split) — expected to look better
    than it should, because windows from the same recording can land in
    both train and test. NOT for reporting.
  - Honest accuracy (StratifiedGroupKFold, grouped by group_id = source
    file) — the number that belongs in the report.

Saves the fitted pipeline + label metadata for reuse in later scripts.
"""

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

from config import OUT_DIR, RANDOM_STATE


DROP_COLS = [
    "session", "file_name", "relative_path", "point_id", "x", "y",
    "group_id", "window_start", "window_end", "grid_label",
]


def build_pipeline(num_cols, cat_cols):
    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("model", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced")),
    ])


def main():
    df = pd.read_csv(OUT_DIR / "feature_table.csv")
    df["grid_label"] = df["place"].astype(str) + "_P" + df["point_id"].astype(int).astype(str).str.zfill(2)

    X = df.drop(columns=DROP_COLS, errors="ignore")
    y = df["grid_label"]
    groups = df["group_id"]

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = ["place"]

    print("X shape:", X.shape, "| Classes:", y.nunique())

    # --- Sanity-check split ------------------------------------------------
    pipe = build_pipeline(num_cols, cat_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    print("\n--- Sanity-check (random split, NOT for reporting) ---")
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Macro-F1:", f1_score(y_test, pred, average="macro"))
    print(classification_report(y_test, pred, zero_division=0))

    # --- Honest grouped split -----------------------------------------------
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    acc_scores, f1_scores = [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        pipe = build_pipeline(num_cols, cat_cols)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = pipe.predict(X.iloc[test_idx])

        acc = accuracy_score(y.iloc[test_idx], pred)
        f1 = f1_score(y.iloc[test_idx], pred, average="macro")
        acc_scores.append(acc)
        f1_scores.append(f1)
        print(f"Fold {fold}: Accuracy={acc:.3f}  Macro-F1={f1:.3f}")

    print("\n--- Honest (grouped by source file) ---")
    print(f"Mean Accuracy: {np.mean(acc_scores):.3f} +/- {np.std(acc_scores):.3f}")
    print(f"Mean Macro-F1: {np.mean(f1_scores):.3f} +/- {np.std(f1_scores):.3f}")

    # Save a pipeline fit on all data for downstream scripts (importance, artifact reuse)
    final_pipe = build_pipeline(num_cols, cat_cols)
    final_pipe.fit(X, y)
    joblib.dump(final_pipe, OUT_DIR / "rf_model.pkl")
    print(f"\nSaved fitted pipeline: {OUT_DIR / 'rf_model.pkl'}")


if __name__ == "__main__":
    main()
