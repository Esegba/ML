"""
09_data_coverage_report.py

Quantifies how many independent recording sessions back each grid point.
This is the concrete data behind the argument that accuracy is capped by
data collection, not modeling: a class with only one session has no way
to demonstrate it generalizes across sessions - there's only one sample
of "session variation" to learn from, let alone test on.

Run after 01_build_feature_table.py. Saves coverage_report.csv and prints
a plain-language summary you can drop straight into a limitations section.
"""

import pandas as pd

from config import OUT_DIR


def main():
    df = pd.read_csv(OUT_DIR / "feature_table.csv")
    df["grid_label"] = df["place"].astype(str) + "_P" + df["point_id"].astype(int).astype(str).str.zfill(2)

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

    lines = []
    lines.append("DATA COVERAGE REPORT")
    lines.append("=" * 40)
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

    lines.append(
        "Recommendation: collect at least 2-3 additional independent recording "
        "sessions per grid point, holding sensor height, orientation, and physical "
        "placement consistent across sessions. This directly targets the model's "
        "actual bottleneck - it is not a hyperparameter problem. A modest accuracy "
        "drop from adding this unseen-session data is expected and is the correct "
        "outcome: it means the reported number is finally measuring generalization "
        "rather than session-specific memorization."
    )

    summary_text = "\n".join(lines)
    print(summary_text)

    with open(OUT_DIR / "coverage_report.txt", "w") as f:
        f.write(summary_text)
    print(f"\nSaved: {OUT_DIR / 'coverage_report.csv'} and coverage_report.txt")


if __name__ == "__main__":
    main()
