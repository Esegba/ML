"""
Helper functions shared across the pipeline scripts:
- filename/folder parsing (place, session, point_id, altitude, grid x/y)
- window-level statistical feature extraction
"""

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dataset_extracted(zip_path: Path, project_root: Path, data_root: Path) -> None:
    """
    Extract the CSV dataset from the Zenodo download if it isn't already
    extracted. Handles the fact that the Zenodo zip is a zip-of-zips:

        18829729.zip
          |- dataset_clean_spike_removed.zip   <- the actual CSVs, wanted
          |- analysis_scripts.zip              <- not needed here

    Safe to call every run: if data_root already has CSVs, it does nothing.
    """
    if data_root.exists() and any(data_root.rglob("*.csv")):
        print(f"Dataset already extracted at {data_root} - skipping extraction.")
        return

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Could not find {zip_path}. Download the dataset from Zenodo "
            f"(DOI 10.5281/zenodo.18829729) and place it there, or update "
            f"DATA_ZIP in config.py."
        )

    print(f"Extracting dataset from {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as outer_zip:
        inner_name = next(
            (n for n in outer_zip.namelist() if n.endswith(".zip") and "dataset_clean" in n.lower()),
            None,
        )

        if inner_name:
            # zip-of-zips: pull the inner dataset zip into memory, then extract it
            inner_bytes = outer_zip.read(inner_name)
            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_zip:
                inner_zip.extractall(project_root)
        else:
            # outer zip already contains dataset_clean/... directly
            outer_zip.extractall(project_root)

    n_csvs = len(list(data_root.rglob("*.csv")))
    if n_csvs == 0:
        raise RuntimeError(
            f"Extraction finished but no CSVs were found under {data_root}. "
            f"Check the zip's internal folder structure."
        )
    print(f"Extracted {n_csvs} CSV files to {data_root}")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize inconsistent column names like 'B norm' -> 'Bnorm'."""
    rename = {}
    for c in df.columns:
        key = c.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
        if key == "bx":
            rename[c] = "Bx"
        elif key == "by":
            rename[c] = "By"
        elif key == "bz":
            rename[c] = "Bz"
        elif key in ["bnorm", "bmag", "btotal", "magnitude", "norm"]:
            rename[c] = "Bnorm"
    return df.rename(columns=rename)


def extract_point_id(file_name: str):
    """Parse a grid point number from filenames like 'ah1', 'ah 2', '_01_', or '(25)'."""
    s = Path(file_name).stem.lower()

    m = re.search(r"\((\d{1,2})\)", s)
    if m:
        return int(m.group(1))

    m = re.search(r"ah\s*(\d{1,2})", s)
    if m:
        return int(m.group(1))

    nums = re.findall(r"(?<!\d)(\d{1,2})(?!\d)", s)
    nums = [int(n) for n in nums if 1 <= int(n) <= 25]
    return nums[-1] if nums else np.nan


def parse_altitude_m(text: str):
    """Parse altitude from session names like '130cm', '150cm', or '1m'."""
    s = str(text).lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*cm", s)
    if m:
        return float(m.group(1)) / 100.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*m\b", s)
    if m:
        return float(m.group(1))
    return np.nan


def point_to_xy(place: str, point_id):
    """
    Map a point number to a local grid (x, y).
    Smythe Park uses a 5x5 grid; City Hall and Scotiabank use a 3x2 grid.
    Confirm this assumption with the project team before using regression
    results in the paper.
    """
    if pd.isna(point_id):
        return np.nan, np.nan
    p = int(point_id)
    cols = 5 if "smythe" in place.lower() else 3
    x = (p - 1) % cols
    y = (p - 1) // cols
    return x, y


def feature_block(series: pd.Series) -> dict:
    """Compute a standard set of summary stats for one sensor axis in a window."""
    v = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(v) == 0:
        return {k: np.nan for k in
                ["mean", "std", "min", "max", "range", "median", "q25", "q75", "cv", "slope"]}

    x = np.arange(len(v))
    mean = np.mean(v)
    std = np.std(v, ddof=1) if len(v) > 1 else 0.0
    slope = np.polyfit(x, v, 1)[0] if len(v) > 1 else 0.0

    return {
        "mean": mean,
        "std": std,
        "min": np.min(v),
        "max": np.max(v),
        "range": np.max(v) - np.min(v),
        "median": np.median(v),
        "q25": np.quantile(v, 0.25),
        "q75": np.quantile(v, 0.75),
        "cv": std / abs(mean) if mean != 0 else np.nan,
        "slope": slope,
    }


def extract_features_window(df_window: pd.DataFrame) -> dict:
    """Build the full feature vector for one window: per-axis stats + cross-axis correlations."""
    feats = {}
    for col in ["Bx", "By", "Bz", "Bnorm"]:
        stats = feature_block(df_window[col])
        for k, v in stats.items():
            feats[f"{col}_{k}"] = v

    for a, b in [("Bx", "By"), ("Bx", "Bz"), ("By", "Bz")]:
        if df_window[a].std() == 0 or df_window[b].std() == 0:
            feats[f"{a}_{b}_corr"] = 0.0
        else:
            feats[f"{a}_{b}_corr"] = df_window[a].corr(df_window[b])

    return feats
