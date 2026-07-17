"""
07_summarize_findings.py

Pulls together the saved results from the earlier scripts into one
plain-text summary. Run last, after 01, 03, and 05.
"""

import pandas as pd

from config import OUT_DIR


def main():
    ml_df = pd.read_csv(OUT_DIR / "feature_table.csv")
    lines = []
    lines.append("QUANTUM SENSOR ML PIPELINE - SUMMARY OF FINDINGS")
    lines.append("=" * 55)
    lines.append(f"Total windows: {len(ml_df)}")
    lines.append(
        f"Places: {ml_df['place'].nunique()}  |  Sessions: {ml_df['session'].nunique()}  "
        f"|  Source files: {ml_df['group_id'].nunique()}"
    )
    lines.append("")

    cls_path = OUT_DIR / "classifier_comparison.csv"
    if cls_path.exists():
        cls_df = pd.read_csv(cls_path)
        lines.append("Classification (grouped, leakage-safe CV):")
        for _, row in cls_df.iterrows():
            lines.append(
                f"  {row['Model']}: Acc={row['Mean Acc']:.3f} +/- {row['Std Acc']:.3f}, "
                f"Macro-F1={row['Mean Macro-F1']:.3f}"
            )
    else:
        lines.append("Classification results not found - run 03_compare_classifiers.py first.")
    lines.append("")

    reg_path = OUT_DIR / "regression_results.csv"
    if reg_path.exists():
        reg_df = pd.read_csv(reg_path)
        row = reg_df.iloc[0]
        lines.append("Regression (file-grouped split):")
        lines.append(f"  MAE={row['MAE']:.3f}, RMSE={row['RMSE']:.3f}, R2={row['R2']:.3f}")
    else:
        lines.append("Regression results not found - run 05_train_regression.py first.")
    lines.append("")

    lines.append(
        "Note: sanity-check (random-split) numbers printed by scripts 02 and 05 are "
        "expected to look better than the grouped numbers above - that gap is the "
        "leakage effect, not a real capability difference. Report the grouped numbers."
    )

    summary_text = "\n".join(lines)
    print(summary_text)

    out_path = OUT_DIR / "summary_findings.txt"
    with open(out_path, "w") as f:
        f.write(summary_text)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
