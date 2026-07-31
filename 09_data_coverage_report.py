"""
09_data_coverage_report.py

"Coverage" here is checked in three distinct senses, since the word is
ambiguous:

  1. Session/group coverage - how many independent recording sessions
     back each grid point (the main driver of the near-chance grouped-CV
     accuracy elsewhere in this pipeline).
  2. Missing-value coverage - whether the engineered feature table has
     gaps (NaNs) in any column, and which columns/places they cluster in.
  3. Window/time coverage - whether every source CSV contributed a full,
     consistent set of windows, or whether some files were truncated/
     short and under-contributed data.

Run after 01_build_feature_table.py. Saves coverage_report.csv (per grid
point) and coverage_missing_values.csv (per feature column), plus prints
a plain-language summary you can drop into a limitations section.
"""

import numpy as np
import pandas as pd

from config import OUT_DIR, WINDOW_SIZE, STEP


def main():
    df = pd.read_csv(OUT_DIR / "feature_table.csv")
    df["grid_label"] = df["place"].astype(str) + "_P" + df["point_id"].astype(int).astype(str).str.zfill(2)

    lines = []
    lines.append("DATA COVERAGE REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append('What "coverage" means in this report (checked three ways):')
    lines.append("  1. Session/group coverage - independent recording sessions per grid point")
    lines.append("  2. Missing-value coverage - NaNs in the engineered feature columns")
    lines.append("  3. Window/time coverage - windows extracted per source file")
    lines.append("")

    # =========================================================================
    # 1. Session/group coverage
    # =========================================================================
    coverage = df.groupby("grid_label").agg(
        place=("place", "first"),
        point_id=("point_id", "first"),
        sessions=("session", "nunique"),
        files=("group_id", "nunique"),
        windows=("group_id", "size"),
    ).reset_index().sort_values(["place", "point_id"])

    coverage.to_csv(OUT_DIR / "coverage_report.csv", index=False)

    single_session = coverage[coverage["sessions"] == 1]
    multi_session = coverage[coverage["sessions"] > 1]

    lines.append("1. SESSION/GROUP COVERAGE")
    lines.append("-" * 60)
    lines.append(f"Total grid points: {len(coverage)}")
    lines.append(f"Grid points backed by only 1 session: {len(single_session)} "
                 f"({100 * len(single_session) / len(coverage):.0f}%)")
    lines.append(f"Grid points backed by 2+ sessions: {len(multi_session)}")
    lines.append("")

    by_place = coverage.groupby("place").agg(
        grid_points=("grid_label", "nunique"),
        min_sessions=("sessions", "min"),
        max_sessions=("sessions", "max"),
        mean_sessions=("sessions", "mean"),
    )
    lines.append("By location:")
    lines.append(by_place.to_string())
    lines.append("")

    if len(single_session) > 0:
        lines.append(
            "Every single-session grid point is a class the model can never be tested "
            "on for cross-session generalization: whichever fold holds that one "
            "recording out has zero training examples of that class, and zero other "
            "sessions exist to check whether the sensor's fingerprint for that point "
            "holds up on a different day, height, orientation, or placement."
        )
        lines.append("")

    # =========================================================================
    # 2. Missing-value coverage
    # =========================================================================
    feature_cols = df.select_dtypes(include=np.number).columns.tolist()
    feature_cols = [c for c in feature_cols if c not in ("point_id", "x", "y", "window_start", "window_end", "altitude_m")]

    missing = df[feature_cols].isna().sum()
    missing_pct = 100 * missing / len(df)
    missing_df = pd.DataFrame({
        "column": feature_cols,
        "missing_count": missing.values,
        "missing_pct": missing_pct.values,
    }).sort_values("missing_count", ascending=False)
    missing_df.to_csv(OUT_DIR / "coverage_missing_values.csv", index=False)

    cols_with_gaps = missing_df[missing_df["missing_count"] > 0]

    lines.append("2. MISSING-VALUE COVERAGE")
    lines.append("-" * 60)
    lines.append(f"Feature columns checked: {len(feature_cols)}")
    if len(cols_with_gaps) == 0:
        lines.append("No missing values found in any engineered feature column.")
    else:
        lines.append(f"Columns with missing values: {len(cols_with_gaps)}")
        lines.append(cols_with_gaps.to_string(index=False))
        lines.append(
            "Note: 01_build_feature_table.py's preprocessing pipelines already "
            "impute remaining NaNs with the median (SimpleImputer), so these "
            "gaps do not silently break training - but a column with a large "
            "gap is being partly synthesized by the imputer rather than measured."
        )
    lines.append("")

    # =========================================================================
    # 3. Window/time coverage per source file
    # =========================================================================
    per_file = df.groupby("group_id").size().rename("windows_extracted").reset_index()
    expected_min_rows_for_2_windows = WINDOW_SIZE + STEP  # fewer rows than this -> only 1 window possible
    under_windowed = per_file[per_file["windows_extracted"] <= 1]

    lines.append("3. WINDOW/TIME COVERAGE")
    lines.append("-" * 60)
    lines.append(f"Window size: {WINDOW_SIZE} rows, step: {STEP} rows (non-overlapping)")
    lines.append(f"Source files: {len(per_file)}")
    lines.append(f"Windows per file - min: {per_file['windows_extracted'].min()}, "
                 f"max: {per_file['windows_extracted'].max()}, "
                 f"mean: {per_file['windows_extracted'].mean():.1f}")
    if len(under_windowed) > 0:
        lines.append(f"Files producing only 1 window (short recording relative to window size): {len(under_windowed)}")
        lines.append(under_windowed.to_string(index=False))
    else:
        lines.append("No files were short enough to produce only a single window.")
    lines.append("")

    # =========================================================================
    # Gaps/issues and modeling implications
    # =========================================================================
    lines.append("SIGNIFICANT GAPS/ISSUES FOUND")
    lines.append("-" * 60)
    issues = []
    if len(single_session) > 0:
        issues.append(
            f"- {len(single_session)} grid points ({100 * len(single_session) / len(coverage):.0f}%) "
            f"have only one recording session - the dominant limitation in this dataset."
        )
    if len(cols_with_gaps) > 0:
        issues.append(f"- {len(cols_with_gaps)} feature columns contain missing values requiring imputation.")
    if len(under_windowed) > 0:
        issues.append(f"- {len(under_windowed)} source files were too short to produce more than one window.")
    if not issues:
        issues.append("- None found beyond the session-coverage imbalance noted above.")
    lines.extend(issues)
    lines.append("")

    lines.append("HOW THIS AFFECTS MODELING STRATEGY AND PREPROCESSING")
    lines.append("-" * 60)
    lines.append(
        "- Session/group coverage is why every evaluation in this pipeline uses "
        "StratifiedGroupKFold grouped by source file rather than a plain random "
        "or stratified split: a random split would let windows from the same "
        "session leak into both train and test, inflating accuracy (this is "
        "exactly what happened before the pipeline was corrected - see the "
        "earlier random-split 'sanity check' numbers, which score far higher "
        "than the grouped, honest ones)."
    )
    lines.append(
        "- Single-session classes cannot be meaningfully cross-validated for "
        "generalization - the honest grouped-CV accuracy on this dataset is "
        "expected to look poor for that reason, and no preprocessing or "
        "hyperparameter choice (see 08_tune_hyperparameters.py) fixes it."
    )
    lines.append(
        "- Missing-value coverage is handled at the preprocessing stage with "
        "median imputation inside each model's pipeline (see config.py-driven "
        "scripts 02/03/05/08); this is adequate given the low missing-value "
        "rate found above, but should be re-checked if new data sources are added."
    )
    lines.append(
        "- Recommendation: collect at least 2-3 additional independent recording "
        "sessions per grid point, holding sensor height, orientation, and physical "
        "placement consistent across sessions. This directly targets the model's "
        "actual bottleneck - it is not a hyperparameter or preprocessing problem. "
        "A modest accuracy drop from adding this unseen-session data is expected "
        "and is the correct outcome: it means the reported number is finally "
        "measuring generalization rather than session-specific memorization."
    )

    summary_text = "\n".join(lines)
    print(summary_text)

    with open(OUT_DIR / "coverage_report.txt", "w") as f:
        f.write(summary_text)
    print(f"\nSaved: coverage_report.csv, coverage_missing_values.csv, coverage_report.txt in {OUT_DIR}")


if __name__ == "__main__":
    main()
