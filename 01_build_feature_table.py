"""
01_build_feature_table.py

Reads every raw CSV under DATA_ROOT, splits each recording into fixed-size
windows, and extracts a feature vector per window. Saves the result to
outputs/feature_table.csv.

`group_id` (the file's relative path) is carried through untouched — it's
the key every later script groups on to keep train/test splits leakage-safe.
"""

import numpy as np
import pandas as pd

from config import DATA_ZIP, DATA_ROOT, PROJECT_ROOT, OUT_DIR, WINDOW_SIZE, STEP
from utils import (
    ensure_dataset_extracted, standardize_columns, extract_point_id,
    parse_altitude_m, point_to_xy, extract_features_window,
)


def main():
    ensure_dataset_extracted(DATA_ZIP, PROJECT_ROOT, DATA_ROOT)

    csv_files = sorted(DATA_ROOT.rglob("*.csv"))
    print(f"Total CSV files found under {DATA_ROOT}: {len(csv_files)}")

    rows = []
    errors = []

    for f in csv_files:
        rel = f.relative_to(DATA_ROOT)
        parts = rel.parts

        place = parts[0]
        session = parts[1] if len(parts) > 1 else "unknown_session"
        point_id = extract_point_id(f.name)
        altitude_m = parse_altitude_m(session)
        x, y = point_to_xy(place, point_id)

        try:
            df = pd.read_csv(f)
            df = standardize_columns(df)

            if "Bnorm" not in df.columns:
                df["Bnorm"] = np.sqrt(df["Bx"] ** 2 + df["By"] ** 2 + df["Bz"] ** 2)

            needed = ["Bx", "By", "Bz", "Bnorm"]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                errors.append((str(rel), missing))
                continue

            for start in range(0, len(df) - WINDOW_SIZE + 1, STEP):
                w = df.iloc[start:start + WINDOW_SIZE].copy()
                feats = extract_features_window(w)
                feats.update({
                    "place": place,
                    "session": session,
                    "file_name": f.name,
                    "relative_path": str(rel),
                    "point_id": int(point_id),
                    "x": x,
                    "y": y,
                    "altitude_m": altitude_m,
                    "window_start": start,
                    "window_end": start + len(w),
                    "group_id": str(rel),
                })
                rows.append(feats)

        except Exception as e:
            errors.append((str(rel), str(e)))

    ml_df = pd.DataFrame(rows)
    print("ML dataset shape:", ml_df.shape)
    print("Files with errors:", len(errors))
    if errors:
        print("First few errors:", errors[:5])

    summary = ml_df.groupby("place").agg(
        windows=("file_name", "size"),
        files=("file_name", "nunique"),
        grid_points=("point_id", "nunique"),
        sessions=("session", "nunique"),
    )
    print("\nSummary by place:")
    print(summary)

    out_path = OUT_DIR / "feature_table.csv"
    ml_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
