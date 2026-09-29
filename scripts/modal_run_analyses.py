"""Run the registered analyses on Modal, where `lme4` fits H6.

Thin orchestration: every number comes from the scripts in `paper/analyses`, unchanged.
The laptop's R segfaults on H6's cell table, so the pipeline runs in an environment whose
R does not, and the launcher - which is the machine that holds the repository - verifies
the tree is clean and passes the commit it verified.

The snapshot lives on its own volume and is never rebuilt: the loader hashes it and
refuses to return data unless it reproduces the primary analysis exactly, so running it
here is only safe if that check passes, which is what `verify` is for.

    modal run scripts/modal_run_analyses.py::verify          # the loader, and nothing else
    modal run --detach scripts/modal_run_analyses.py::run    # S11 and S12
"""

import os

import modal

ANALYSES = os.path.join(os.path.expanduser("~"),
                        "Documents/GitHub/ecological-bias-covid/paper/analyses")
REMOTE = "/root/repo/paper/analyses"

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("r-base-core", "r-cran-lme4", "r-cran-jsonlite", "r-cran-matrix")
    .run_commands(
        "R -e \"stopifnot(requireNamespace('lme4'), requireNamespace('jsonlite')); "
        "cat(R.version.string, as.character(packageVersion('lme4')))\"",
    )
    .pip_install(
        "numpy==2.5.1",
        "scipy==1.18.0",
        "pandas==3.0.3",
        "patsy==1.0.2",
        "statsmodels==0.14.6",
    )
    .env({"PYTHONPATH": REMOTE})
    .add_local_dir(ANALYSES, REMOTE, copy=True,
                   ignore=["__pycache__", "results", "tests", "logs"])
)

app = modal.App("jamia-analyses", image=image)
snapshot = modal.Volume.from_name("jamia-mexico-snapshot")
results = modal.Volume.from_name("jamia-analysis-results", create_if_missing=True)

DATA = "/root/repo/data/mexico_covid"
OUT = "/root/repo/paper/analyses/results"
TIMEOUT = 86_400


def _prepare(commit):
    """Record who verified the commit, and make sure results have somewhere to land.

    The snapshot volume carries the CSV beside the primary analysis the loader checks
    itself against, mounted at the path the loader already expects.
    """
    import os
    from pathlib import Path

    os.environ["ANALYSIS_COMMIT"] = commit
    Path(OUT).mkdir(parents=True, exist_ok=True)


@app.function(cpu=4.0, memory=32_768, timeout=TIMEOUT,
              volumes={DATA: snapshot, OUT: results})
def verify_loader(commit: str) -> dict:
    """Does the snapshot reproduce the primary analysis in this environment?"""
    import json

    _prepare(commit)
    import mexico_confirmed_cases

    frame = mexico_confirmed_cases.load()
    results.commit()
    audit = json.load(open(f"{OUT}/mexico_loader_audit.json"))
    return {
        "records": int(len(frame)),
        "deaths": int(frame["died"].sum()),
        "treating_states": int(frame["site"].nunique()),
        "counts": audit.get("counts"),
        "outside_catalogue": audit.get("columns", {}).get("outside_catalogue"),
    }


@app.function(cpu=8.0, memory=65_536, timeout=TIMEOUT,
              volumes={DATA: snapshot, OUT: results})
def run_stage(stage: str, commit: str) -> str:
    """One registered analysis, writing its result file to the results volume."""
    from datetime import datetime, timezone

    _prepare(commit)
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {stage} starting", flush=True)
    if stage == "s11":
        import s11_site_definitions as analysis
    elif stage == "s12":
        import s12_composition_ladder as analysis
    else:
        raise ValueError(f"no stage named {stage}")
    analysis.main()
    results.commit()
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {stage} done", flush=True)
    return stage


def _verified_commit():
    """The launcher does the check the container cannot: a clean, committed tree."""
    from paths import _git, require_clean_tree

    require_clean_tree()
    return _git("rev-parse", "HEAD").stdout.strip()


@app.local_entrypoint()
def verify():
    import json

    print(json.dumps(verify_loader.remote(_verified_commit()), indent=2))


@app.local_entrypoint()
def run(stages: str = "s11,s12"):
    commit = _verified_commit()
    print(f"running {stages} from {commit[:12]}")
    for stage in run_stage.map(stages.split(","), kwargs={"commit": commit},
                               return_exceptions=True):
        print(f"  {stage}", flush=True)
