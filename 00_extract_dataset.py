"""
00_extract_dataset.py

Extracts the CSV dataset from the Zenodo download (DOI 10.5281/zenodo.18829729)
into dataset_clean/ under PROJECT_ROOT. The Zenodo zip is a zip-of-zips:

    18829729.zip
      |- dataset_clean_spike_removed.zip   <- the CSVs (what we want)
      |- analysis_scripts.zip              <- not needed by this pipeline

This step also runs automatically inside 01_build_feature_table.py, so you
don't have to run it separately - it's here in case you just want to unpack
the data and look at it first.
"""

from config import DATA_ZIP, DATA_ROOT, PROJECT_ROOT
from utils import ensure_dataset_extracted


def main():
    ensure_dataset_extracted(DATA_ZIP, PROJECT_ROOT, DATA_ROOT)


if __name__ == "__main__":
    main()
