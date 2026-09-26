"""The one place a location is computed from __file__."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT_ROOT / "data"
MEXICO_DIR = DATA / "mexico_covid"
MEXICO_SNAPSHOT_CSV = MEXICO_DIR / "220103COVID19MEXICO.csv"
MEXICO_MAIN_RESULTS = MEXICO_DIR / "mexico_ecological_results.json"
RESULTS = PROJECT_ROOT / "paper" / "analyses" / "results"
