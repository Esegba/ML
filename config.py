"""
Shared paths and constants for the quantum sensor ML pipeline.

Edit PROJECT_ROOT if your folder ever moves. Everything else derives from it,
so you only change one line.
"""

from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\esegb\OneDrive\Desktop\ML")

# The Zenodo download (DOI 10.5281/zenodo.18829729). Place it here, or edit
# this path to wherever you saved it. It's a zip-of-zips: this outer file
# contains "dataset_clean_spike_removed.zip" (the CSVs) and
# "analysis_scripts.zip" (not needed by this pipeline).
DATA_ZIP = PROJECT_ROOT / "18829729.zip"

DATA_ROOT = PROJECT_ROOT / "dataset_clean"   # raw per-place / per-session CSVs (extracted from DATA_ZIP)
OUT_DIR = PROJECT_ROOT / "outputs"           # everything this pipeline produces goes here

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Window feature extraction
WINDOW_SIZE = 100
STEP = 100

# Reproducibility
RANDOM_STATE = 42
