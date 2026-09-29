"""The one place a location is computed from __file__, and the one way a result is written."""

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT_ROOT / "data"
MEXICO_DIR = DATA / "mexico_covid"
MEXICO_SNAPSHOT_CSV = MEXICO_DIR / "220103COVID19MEXICO.csv"
MEXICO_MAIN_RESULTS = MEXICO_DIR / "mexico_ecological_results.json"
RESULTS = PROJECT_ROOT / "paper" / "analyses" / "results"
EXPECTED_SNAPSHOT_SHA256 = "25ccc890d190bf66a90aae98507f78f6bfe8b0e9be10767936a0e929512dbc6d"


def archive_existing(path):
    """Move a file that already exists into results/superseded/, never overwriting.

    The archived name carries the timestamp the file itself declares, or its
    modification time, to microseconds, plus a counter if that name is taken.
    """
    path = Path(path)
    if not path.exists():
        return None
    superseded = path.parent / "superseded"
    superseded.mkdir(exist_ok=True)
    stamp = ""
    if path.suffix == ".json":
        try:
            stamp = re.sub(r"[^0-9]", "", json.loads(path.read_text()).get("timestamp", ""))
        except (json.JSONDecodeError, OSError, AttributeError):
            stamp = ""
    if not stamp:
        stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y%m%d%H%M%S%f")
    target = superseded / f"{path.stem}_{stamp}{path.suffix}"
    counter = 1
    while target.exists():
        target = superseded / f"{path.stem}_{stamp}_{counter}{path.suffix}"
        counter += 1
    path.rename(target)
    return target


def _replace_atomically(path, write, validate):
    """Write beside the target, check it reads back, archive the old file, then rename.

    The archive step is what keeps provenance; doing it before the new file exists would
    leave no current result if the write were interrupted, so the new file is written and
    validated first and the rename is the only step that cannot half-happen.
    """
    path = Path(path)
    pending = path.with_name(path.name + ".pending")
    try:
        write(pending)
        validate(pending)
        archive_existing(path)
        pending.replace(path)
    finally:
        if pending.exists():
            pending.unlink()
    return path


def write_result(payload, path):
    """Write a result file, archiving any existing one rather than overwriting it."""
    return _replace_atomically(
        path,
        lambda target: target.write_text(json.dumps(payload, indent=2)),
        lambda target: json.loads(target.read_text()))


def write_table(frame, path):
    """Write a sidecar CSV, archiving any existing one rather than overwriting it."""
    return _replace_atomically(
        path,
        lambda target: frame.to_csv(target, index=False),
        lambda target: _first_line(target))


def _first_line(path):
    with open(path) as handle:
        if not handle.readline():
            raise ValueError(f"{path} is empty")


# Every path whose contents can change a reported number: the analyses themselves and the
# wrappers that launch them. A file outside these cannot affect a run.
TRACKED_FOR_RUNS = ("paper/analyses", "scripts")


def _git(*arguments):
    """Run git where it exists; elsewhere say so rather than raising on the binary."""
    try:
        return subprocess.run(["git", "-C", str(PROJECT_ROOT), *arguments],
                              capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return subprocess.CompletedProcess(arguments, returncode=127, stdout="",
                                           stderr="git is not installed here")


def require_clean_tree():
    """Refuse to produce a reported result from an uncommitted analysis tree."""
    status = _git("status", "--porcelain", *TRACKED_FOR_RUNS)
    if status.returncode != 0:
        raise RuntimeError(f"git could not report the state of {PROJECT_ROOT}: "
                           f"{status.stderr.strip()}")
    dirty = status.stdout.strip()
    if dirty:
        raise RuntimeError(
            "the analysis tree has uncommitted changes, so the run could not be traced to "
            f"a commit; commit them first:\n{dirty}")


def environment():
    """The versions and source hashes a rerun would have to match to reproduce a number."""
    modules = {}
    for name in ("numpy", "scipy", "pandas", "patsy", "statsmodels"):
        try:
            modules[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            modules[name] = "absent"
    analyses = PROJECT_ROOT / "paper" / "analyses"
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()[:16]
               for path in sorted(analyses.glob("*.py"))}
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": modules,
        "analysis_source_sha256_prefixes": sources,
    }


def run_metadata(run_id):
    """What every output of one run carries: the run, the code, the data and the clock."""
    commit = _git("rev-parse", "HEAD").stdout.strip()
    status = _git("status", "--porcelain", *TRACKED_FOR_RUNS)
    dirty = status.stdout.strip()
    return {
        "run_id": run_id,
        "commit": commit or "unknown",
        "analysis_tree_dirty": bool(dirty) if status.returncode == 0 else None,
        "paths_watched_for_changes": list(TRACKED_FOR_RUNS),
        "snapshot_sha256": EXPECTED_SNAPSHOT_SHA256,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "environment": environment(),
    }
