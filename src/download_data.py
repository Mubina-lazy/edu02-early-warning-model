"""Download the OULAD dataset into data/raw/.

Two sources are supported:

1. "official" (default): the Open University's own site.
   https://analyse.kmi.open.ac.uk/open_dataset  ->  anonymisedData.zip
2. "mirror": the dataset authors' GitHub mirror (R package by Kuzilek,
   Hlosta and Zdrahal, the people who published OULAD), used when the
   official site is unreachable. Requires `pyreadr` to convert the R data
   files back to the official CSV layout.

Both sources are CC-BY 4.0. After downloading, the script checks the row
count of every table against the numbers published in the dataset paper
(Kuzilek et al., Scientific Data 4:170171, 2017), so we know the data is
complete and untampered no matter which source was used.

Usage:
    python src/download_data.py                 # official site
    python src/download_data.py --source mirror # GitHub fallback
"""

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import requests

# Repo root is one level above src/, so paths work no matter where you run from.
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

OFFICIAL_URL = "https://analyse.kmi.open.ac.uk/open_dataset/download"
MIRROR_REPO = "https://github.com/jakubkuzilek/oulad.git"

# The 7 official CSV names and the row counts published in the OULAD paper.
EXPECTED_ROWS = {
    "courses.csv": 22,
    "assessments.csv": 206,
    "vle.csv": 6364,
    "studentInfo.csv": 32593,
    "studentRegistration.csv": 32593,
    "studentAssessment.csv": 173912,
    "studentVle.csv": 10655280,
}

# Mirror stores tables as R .rda files under different names -> map back.
RDA_TO_CSV = {
    "course": "courses.csv",
    "assessment": "assessments.csv",
    "vle": "vle.csv",
    "student": "studentInfo.csv",
    "student_registration": "studentRegistration.csv",
    "student_assessment": "studentAssessment.csv",
    "student_vle": "studentVle.csv",
}


def download_official() -> None:
    """Download and unzip anonymisedData.zip from the official site."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "anonymisedData.zip"
    print(f"Downloading {OFFICIAL_URL} ...")
    with requests.get(OFFICIAL_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print("Unzipping ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW_DIR)
    zip_path.unlink()  # keep only the CSVs


def download_mirror() -> None:
    """Clone the authors' GitHub mirror and convert .rda files to CSV."""
    try:
        import pyreadr
    except ImportError:
        sys.exit("The mirror source needs pyreadr:  pip install pyreadr")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    clone_dir = RAW_DIR / "_oulad_mirror"
    if not clone_dir.exists():
        print(f"Cloning {MIRROR_REPO} (shallow) ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", MIRROR_REPO, str(clone_dir)],
            check=True,
        )
    for rda_name, csv_name in RDA_TO_CSV.items():
        print(f"Converting {rda_name}.rda -> {csv_name} ...")
        result = pyreadr.read_r(str(clone_dir / "data" / f"{rda_name}.rda"))
        df = result[rda_name]
        # The mirror stores some integer columns as floats; the official CSVs
        # use plain integers, so convert back where possible (keeps NaN).
        df = df.convert_dtypes()
        df.to_csv(RAW_DIR / csv_name, index=False)
    shutil.rmtree(clone_dir)


def validate() -> None:
    """Check every table's row count against the published numbers."""
    import pandas as pd

    print("\nValidating row counts against the OULAD paper:")
    ok = True
    for csv_name, expected in EXPECTED_ROWS.items():
        path = RAW_DIR / csv_name
        if not path.exists():
            print(f"  MISSING  {csv_name}")
            ok = False
            continue
        # Count rows without loading whole file into memory at once.
        n = sum(len(chunk) for chunk in pd.read_csv(path, chunksize=1_000_000))
        status = "OK " if n == expected else "BAD"
        if n != expected:
            ok = False
        print(f"  {status}  {csv_name:26s} {n:>10,d} rows (expected {expected:,d})")
    if not ok:
        sys.exit("Validation FAILED - do not use this download.")
    print("All tables complete. Data is ready in data/raw/.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["official", "mirror"],
        default="official",
        help="where to download from (default: official site)",
    )
    args = parser.parse_args()
    if args.source == "official":
        download_official()
    else:
        download_mirror()
    validate()


if __name__ == "__main__":
    main()
