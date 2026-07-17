# Quantum Sensor ML Pipeline (VS Code version)

Merged from the Colab notebook and the GitHub repo scripts, restructured as
plain `.py` files for local development in VS Code.

## Setup

```
cd C:\Users\esegb\OneDrive\Desktop\ML_Project
.venv\Scripts\activate
pip install -r requirements.txt
```

Place the Zenodo download (`18829729.zip`, DOI 10.5281/zenodo.18829729) in
the project root, or edit `DATA_ZIP` in `config.py` to point wherever you
saved it. Update `PROJECT_ROOT` too if the whole folder ever moves —
everything else derives from those two paths.

The zip is a zip-of-zips (`dataset_clean_spike_removed.zip` +
`analysis_scripts.zip` inside it) — `00_extract_dataset.py` /
`01_build_feature_table.py` unwrap it automatically into `dataset_clean/`.
Already-extracted data is detected and skipped, so it's safe to re-run.

## Run order

```
python 00_extract_dataset.py     # optional - 01 also runs this automatically
python 01_build_feature_table.py
python 02_train_classifier.py
python 03_compare_classifiers.py
python 04_evaluate_classifier.py
python 05_train_regression.py
python 06_visualize.py
python 07_summarize_findings.py
```

Each script reads what it needs from `outputs/` and writes its results back
there (`feature_table.csv`, `.pkl` models, `.csv` results, `.png` plots,
`summary_findings.txt`).

## What changed from the original repo

The original `10_compare_classifiers.py` ran a plain `StratifiedKFold` on a
single session's `features_raw.csv` — the same leakage pattern flagged
earlier (windows from one recording split across train and test), and it's
what produced the 99.9-100% "too good to be true" accuracy.

`03_compare_classifiers.py` here uses `StratifiedGroupKFold` grouped by
`group_id` (the source file), matching the honest split already used in
`02_train_classifier.py`. Every evaluation in this pipeline is
leakage-safe now, not just some of them.

`04_evaluate_classifier.py`'s confusion matrix is built from grouped
out-of-fold predictions rather than the fitted model re-predicting data
it was trained on, which is what the original repo script did (a
memorization baseline, not a test result).

Each script still prints a "sanity-check" random-split number alongside
the honest one, so you can see the size of the leakage effect directly —
just don't report the sanity-check number.
