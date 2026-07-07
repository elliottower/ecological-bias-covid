"""
Download 4CE neurological phenotype data from GitHub.

The Phase2.1NeuroAnalysis repo contains site-level .rda files with aggregate
results from 21 healthcare systems across 6 countries. Each file contains
summary statistics (no patient-level data) for neurological outcomes in
COVID-19 patients.
"""

from pathlib import Path

import requests
from tqdm import tqdm

REPO = "covidclinical/Phase2.1NeuroAnalysis"
BRANCH = "master"
RESULTS_DIRS = ["results", "results_comorbidity"]
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"

KNOWN_SITES = [
    "APHP", "BCH", "FRBDX", "GOSH", "H12O", "HPG23",
    "ICSM", "MGB", "NUH", "NWU", "UCLA", "UKFR",
    "UKY", "UMICH", "UPENN", "UPITT",
    "VA1", "VA2", "VA3", "VA4", "VA5",
]


def download_4ce_rda_files(output_dir: Path) -> list[Path]:
    """Download .rda result files from the 4CE neuro analysis repo."""
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for results_dir in RESULTS_DIRS:
        subdir = output_dir / results_dir
        subdir.mkdir(parents=True, exist_ok=True)

        for site in tqdm(KNOWN_SITES, desc=f"Downloading {results_dir}"):
            if results_dir == "results_comorbidity":
                filename = f"{site}_comorb_results.rda"
            else:
                filename = f"{site}_results.rda"
            url = f"{RAW_URL}/{results_dir}/{filename}"

            out_path = subdir / filename
            if out_path.exists():
                downloaded.append(out_path)
                continue

            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    out_path.write_bytes(resp.content)
                    downloaded.append(out_path)
                else:
                    print(f"  {filename}: HTTP {resp.status_code}")
            except requests.RequestException as e:
                print(f"  {filename}: {e}")

    return downloaded


def download_analysis_code(output_dir: Path) -> None:
    """Download the R analysis scripts for reference."""
    scripts = [
        "neuro_metaanalysis_master.Rmd",
        "run_neuro_model.R",
    ]
    code_dir = output_dir / "analysis_code"
    code_dir.mkdir(parents=True, exist_ok=True)

    for script in scripts:
        url = f"{RAW_URL}/{script}"
        out_path = code_dir / script
        if out_path.exists():
            continue
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                out_path.write_text(resp.text)
                print(f"  Downloaded {script}")
        except requests.RequestException:
            pass


def load_rda_to_dict(rda_path: Path) -> dict:
    """Load an .rda file and return its contents as a dict of DataFrames."""
    import pyreadr
    result = pyreadr.read_r(str(rda_path))
    return result


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data" / "4ce_neuro"
    print(f"Downloading 4CE neuro data to {data_dir}")
    files = download_4ce_rda_files(data_dir)
    print(f"\nDownloaded {len(files)} files")
    download_analysis_code(data_dir)

    if files:
        print(f"\nInspecting first file: {files[0]}")
        try:
            data = load_rda_to_dict(files[0])
            for name, df in data.items():
                print(f"  {name}: shape={df.shape}, columns={list(df.columns)[:10]}")
        except Exception as e:
            print(f"  Error reading: {e}")
