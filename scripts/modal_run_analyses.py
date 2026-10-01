"""Run the registered analyses on Modal, where `lme4` fits H6.

Thin orchestration: every number comes from the scripts in `paper/analyses`, unchanged.
The laptop's R segfaults on H6's cell table, so the pipeline runs in an environment whose
R does not, and the launcher - which is the machine that holds the repository - verifies
the tree is clean and passes the commit it verified.

The snapshot lives on its own volume and is never rebuilt: the loader hashes it and
refuses to return data unless it reproduces the primary analysis exactly, so running it
here is only safe if that check passes, which is what `verify` is for.

    modal run scripts/modal_run_analyses.py::verify          # the loader, and nothing else
    modal deploy scripts/modal_run_analyses.py               # then
    modal run scripts/modal_run_analyses.py::launch --stages s12

`launch` spawns on the deployed app, which is the only form that survives the laptop
sleeping: `--detach` keeps the last triggered function alive after the client goes away,
and a deployed function never depended on the client to begin with. `run` holds the
client open for the whole run and is for a stage short enough to watch.
"""

import os

import modal

ANALYSES = os.path.join(os.path.expanduser("~"),
                        "Documents/GitHub/ecological-bias-covid/paper/analyses")
REMOTE = "/root/repo/paper/analyses"
# Never copied, so never in the manifest the container checks itself against.
IGNORED = ["__pycache__", "results", "tests", "logs", "attestations", ".DS_Store"]
VALIDATION = "/root/repo/paper/analyses/validation/glmm_validation.json"

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
        # the primary analysis the loader checks itself against imports it
        "tqdm==4.70.1",
    )
    .env({"PYTHONPATH": REMOTE})
    .add_local_dir(ANALYSES, REMOTE, copy=True, ignore=IGNORED)
    # The results directory is a mounted volume in the container and does not carry the
    # validation artifact, so it travels separately and the preflight is pointed at it.
    .add_local_file(os.path.join(ANALYSES, "results/glmm_validation.json"),
                    VALIDATION, copy=True)
)

app = modal.App("jamia-analyses", image=image)
snapshot = modal.Volume.from_name("jamia-mexico-snapshot")
results = modal.Volume.from_name("jamia-analysis-results", create_if_missing=True)

DATA = "/root/repo/data/mexico_covid"
OUT = "/root/repo/paper/analyses/results"
TIMEOUT = 86_400


def _manifest(root):
    """sha256 of every file that will be copied, keyed by its path inside the image."""
    import hashlib
    from pathlib import Path

    entries = {}
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or any(part in IGNORED for part in path.parts):
            continue
        entries[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return entries


def _prepare(commit, manifest):
    """Prove the copied files are the commit the launcher verified, then let it run.

    Being told a commit is not the same as holding it. The launcher hashes what it is
    about to send; the container hashes what arrived and refuses to run on a difference,
    which closes the gap between "the launcher verified X" and "this executed X".
    """
    import hashlib
    import os
    from pathlib import Path

    mismatched = []
    for relative, expected in sorted(manifest.items()):
        arrived = Path(REMOTE) / relative
        if not arrived.exists():
            mismatched.append(f"{relative}: missing")
        elif hashlib.sha256(arrived.read_bytes()).hexdigest() != expected:
            mismatched.append(f"{relative}: differs")
    if mismatched:
        raise RuntimeError("the container does not hold the verified commit: "
                           + "; ".join(mismatched))

    os.environ["ANALYSIS_COMMIT"] = commit
    os.environ["VALIDATION_ARTIFACT"] = VALIDATION
    Path(OUT).mkdir(parents=True, exist_ok=True)
    print(f"verified {len(manifest)} files against the launcher's manifest", flush=True)


@app.function(cpu=4.0, memory=32_768, timeout=TIMEOUT,
              volumes={DATA: snapshot, OUT: results})
def verify_loader(commit: str, manifest: dict) -> dict:
    """Does the snapshot reproduce the primary analysis in this environment?"""
    import json

    _prepare(commit, manifest)
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
def run_stage(stage: str, commit: str, manifest: dict) -> str:
    """One registered analysis, writing its result file to the results volume."""
    from datetime import datetime, timezone

    _prepare(commit, manifest)
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

    print(json.dumps(verify_loader.remote(_verified_commit(), _manifest(ANALYSES)),
                     indent=2))


@app.local_entrypoint()
def launch(stages: str = "s12"):
    """Spawn on the deployed app and return; the run does not depend on this client."""
    commit, manifest = _verified_commit(), _manifest(ANALYSES)
    print(f"launching {stages} from {commit[:12]}, {len(manifest)} files in the manifest")
    deployed = modal.Function.from_name(app.name, "run_stage")
    for stage in stages.split(","):
        print(f"  {stage}: {deployed.spawn(stage, commit, manifest).object_id}", flush=True)


@app.local_entrypoint()
def run(stages: str = "s11,s12"):
    commit, manifest = _verified_commit(), _manifest(ANALYSES)
    print(f"running {stages} from {commit[:12]}, {len(manifest)} files in the manifest")
    for stage in run_stage.map(stages.split(","),
                               kwargs={"commit": commit, "manifest": manifest},
                               return_exceptions=True):
        print(f"  {stage}", flush=True)
