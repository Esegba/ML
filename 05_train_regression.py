"""
05_train_regression.py

Predicts local (x, y) grid coordinates. Reports both the random-split
sanity check and the file-grouped honest evaluation, same convention as
the classifier scripts.
"""

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import OUT_DIR, RANDOM_STATE

DROP_COLS = [
    "session", "file_name", "relative_path", "point_id", "x", "y",
    "group_id", "window_start", "window_end",
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
        ("model", RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE)),
    ])


def main():
    reg_df = pd.read_csv(OUT_DIR / "feature_table.csv")

    X_reg = reg_df.drop(columns=DROP_COLS, errors="ignore")
    y_reg = reg_df[["x", "y"]]

    num_cols = X_reg.select_dtypes(include=np.number).columns.tolist()
    cat_cols = ["place"]

    # --- Sanity-check split --------------------------------------------------
    rf_reg = build_pipeline(num_cols, cat_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=RANDOM_STATE
    )
    rf_reg.fit(X_train, y_train)
    pred = rf_reg.predict(X_test)

    print("--- Sanity-check regression (random split, NOT for reporting) ---")
    print("MAE:", mean_absolute_error(y_test, pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
    print("R2:", r2_score(y_test, pred))

    # --- Honest file-grouped split -------------------------------------------
    rf_reg = build_pipeline(num_cols, cat_cols)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X_reg, y_reg, groups=reg_df["group_id"]))

    X_train, X_test = X_reg.iloc[train_idx], X_reg.iloc[test_idx]
    y_train, y_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]

    rf_reg.fit(X_train, y_train)
    pred = rf_reg.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    print("\n--- Honest regression (file-grouped split) ---")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)

    pd.DataFrame([{"MAE": mae, "RMSE": rmse, "R2": r2}]).to_csv(
        OUT_DIR / "regression_results.csv", index=False
    )

    # Save test-set predictions for the plotting script
    pred_df = pd.DataFrame(pred, columns=["x_pred", "y_pred"], index=y_test.index)
    compare_df = y_test.join(pred_df)
    compare_df.to_csv(OUT_DIR / "regression_predictions.csv", index=False)

    joblib.dump(rf_reg, OUT_DIR / "rf_regression_model.pkl")
    print(f"\nSaved: regression_results.csv, regression_predictions.csv, rf_regression_model.pkl in {OUT_DIR}")


if __name__ == "__main__":
    main()
