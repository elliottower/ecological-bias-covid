"""Check `glmm.py`, as independently checkpointed units.

The suite is a set of units, each of which fits models and returns one JSON payload.
A unit is written to its own shard the moment it finishes and is never recomputed while
that shard matches the fingerprint of the estimator it tested, so an interrupted run
resumes instead of restarting, and the units can be run anywhere - in one process here,
or one container each on Modal.

The units:

- `s9_against_lme4`      S9's one-stage model, the one mixed model `lme4::glmer` completed
                         on this machine before it began segfaulting, refitted and
                         compared coefficient by coefficient.
- `reduced_design`       The same sixteen-term design at a size `glmer` should finish,
                         fitted by both implementations.
- `h6_shape`             A table of H6's shape - 32 groups, sixteen terms, 160,000 cells -
                         simulated from known parameters and fitted by both, or `glmer`'s
                         failure on it recorded.
- `node_{k}`             The same data at k quadrature nodes, for k in 7, 11, 15, 21, 25.
- `coverage_{...}`       One batch of repeated-sample coverage for one scenario, counting
                         how often the Wald and profile intervals contain a known truth.

Nothing here reads a registered quantity: every unit but `s9_against_lme4` is simulated,
and that one refits a model whose result file already exists.
"""

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

import glmm
from paths import PROJECT_ROOT, RESULTS, run_metadata, write_result

ANALYSES = PROJECT_ROOT / "paper" / "analyses"
CELLS = RESULTS / "mexico_cells.csv"
REFERENCE = RESULTS / "mexico_glmm_fits.json"
OUTPUT = RESULTS / "glmm_validation.json"
SHARDS = RESULTS / "glmm_validation_shards"
GLMER = ANALYSES / "r" / "glmer_reference.R"

TOLERANCE = 1e-4  # agreement in the log odds ratio, well inside the third decimal of an OR
GLMER_TIMEOUT = 21_600  # six hours, after which glmer is recorded as not having finished
NODE_LADDER = [7, 11, 15, 21, 25]
REPORTED_NODES = 15
SEED = 20260926
COVERAGE_REPLICATIONS = 300
COVERAGE_BATCH = 25
PROFILE_PER_BATCH = 13  # about half of each batch, so profiles cover the whole scenario
COVERAGE_SCENARIOS = [
    {"groups": 32, "sigma": 0.35},
    {"groups": 32, "sigma": 0.00},
    {"groups": 12, "sigma": 0.35},
    {"groups": 120, "sigma": 0.35},
]


def fingerprint():
    """The estimator a shard tested; a shard from different code is not reused."""
    return hashlib.sha256((ANALYSES / "glmm.py").read_bytes()).hexdigest()[:16]


def unit_names():
    names = ["s9_against_lme4", "reduced_design"]
    names += [f"node_{nodes:02d}" for nodes in NODE_LADDER]
    for scenario in COVERAGE_SCENARIOS:
        for batch in range(COVERAGE_REPLICATIONS // COVERAGE_BATCH):
            names.append(coverage_name(scenario, batch))
    names.append("h6_shape")  # last: the most expensive unit, so cheap ones land first
    return names


def coverage_name(scenario, batch):
    return f"coverage_g{scenario['groups']:03d}_s{int(scenario['sigma'] * 100):03d}_b{batch:02d}"


def unit_rng(name):
    """One independent generator per unit, so a unit reruns to the same numbers."""
    digest = hashlib.blake2b(name.encode(), digest_size=8).digest()
    return np.random.default_rng([SEED, int.from_bytes(digest, "big")])


def simulate(rng, groups, sigma, terms=16, cells_per_group=5_000, intercept=-4.0):
    """A design of H6's shape: one intercept, `terms - 1` covariates, `groups` clusters."""
    n_cells = groups * cells_per_group
    index = np.repeat(np.arange(groups), cells_per_group)
    design = np.column_stack([np.ones(n_cells), rng.normal(size=(n_cells, terms - 1))])
    beta = np.concatenate([[intercept], rng.normal(0, 0.3, terms - 1)])
    u = rng.normal(0, sigma, groups) if sigma > 0 else np.zeros(groups)
    n = rng.integers(1, 60, n_cells).astype(float)
    probability = 1 / (1 + np.exp(-(design @ beta + u[index])))
    deaths = rng.binomial(n.astype(int), probability).astype(float)
    return design, deaths, n, index, beta


def fit_in_r(design, deaths, n, groups, nodes):
    """The same data through `lme4::glmer`, or the reason it did not finish."""
    frame = pd.DataFrame({"deaths": deaths, "alive": n - deaths, "group": groups})
    for j in range(1, design.shape[1]):
        frame[f"x{j}"] = design[:, j]
    with tempfile.TemporaryDirectory() as folder:
        cells_path, out_path = Path(folder) / "cells.csv", Path(folder) / "fit.json"
        frame.to_csv(cells_path, index=False)
        started = time.time()
        try:
            attempt = subprocess.run(
                ["Rscript", str(GLMER), str(cells_path), str(out_path), str(nodes)],
                capture_output=True, text=True, check=False, timeout=GLMER_TIMEOUT)
            message, code = attempt.stderr.strip()[-600:], attempt.returncode
        except subprocess.TimeoutExpired:
            message, code = f"no result after {GLMER_TIMEOUT} seconds", None
        except FileNotFoundError:
            message, code = "Rscript is not installed in this environment", None
        if out_path.exists():
            return {**json.load(open(out_path)), "completed": True}
    return {"completed": False, "returncode": code, "stderr_tail": message,
            "seconds": round(time.time() - started, 1)}


def compare(fit, reference, truth=None):
    """Python against R, coefficient by coefficient, and both against the truth."""
    out = {
        "python": {"beta": [float(b) for b in fit["beta"]],
                   "se": [float(s) for s in fit["se"]],
                   "sigma": fit["sigma"], "log_likelihood": fit["log_likelihood"],
                   "quadrature_nodes": fit["quadrature_nodes"],
                   "hessian": fit["hessian"],
                   "variance_starts": fit["variance_starts"]},
        "lme4": reference,
    }
    if truth is not None:
        out["true_beta"] = [float(b) for b in truth]
        out["python_max_absolute_error_against_truth"] = float(np.max(np.abs(fit["beta"] - truth)))
        out["python_largest_z_against_truth"] = float(
            np.max(np.abs((fit["beta"] - truth) / fit["se"])))
    if not reference.get("completed"):
        return out
    estimate = np.array(reference["estimate"], dtype=float)
    out["max_absolute_log_or_difference"] = float(np.max(np.abs(fit["beta"] - estimate)))
    out["max_absolute_se_difference"] = float(
        np.max(np.abs(np.array(fit["se"]) - np.array(reference["se"], dtype=float))))
    out["sigma_difference"] = float(abs(fit["sigma"] - reference["random_intercept_sd"]))
    out["log_likelihood_note"] = (
        "the two log-likelihoods are on different additive scales and are not compared: "
        "lme4's includes the binomial normalizing constant, which accounts for most of "
        "the gap, and a further offset we did not reconcile. Evaluating this "
        "implementation's likelihood at lme4's own estimates returns this "
        "implementation's own maximum, so both optimizers reach the same point; the "
        "comparison that carries the validation is coefficients, standard errors and sigma")
    out["agrees"] = bool(out["max_absolute_log_or_difference"] < TOLERANCE)
    if truth is not None:
        out["lme4_max_absolute_error_against_truth"] = float(np.max(np.abs(estimate - truth)))
    return out


def binomial_constant(deaths, n):
    """The normalizing term this implementation's likelihood omits and lme4's carries."""
    return float(np.sum(gammaln(n + 1) - gammaln(deaths + 1) - gammaln(n - deaths + 1)))


def s9_against_lme4():
    """The real-data comparison: the same model, the two implementations."""
    cells = pd.read_csv(CELLS)
    site_age = cells.groupby(["site", "elderly"], as_index=False).agg(
        deaths=("deaths", "sum"), n=("n", "sum"))
    sites = sorted(site_age["site"].unique())
    design = np.column_stack([np.ones(len(site_age)), site_age["elderly"].to_numpy(float)])
    groups = site_age["site"].map({s: i for i, s in enumerate(sites)}).to_numpy()

    fit = glmm.fit_random_intercept(design, site_age["deaths"], site_age["n"], groups)
    python = glmm.wald(fit["beta"][1], fit["se"][1])
    profile = glmm.profile_interval(fit, 1)
    reference = json.load(open(REFERENCE))["s9_one_stage_random_intercept"]["elderly"]
    difference = abs(python["log_or"] - float(np.log(reference["or"])))
    gradient = glmm.gradient_check(fit)

    return {
        "model": "logit P(death) = intercept + elderly + (1 | treating-unit state)",
        "n_cells": int(len(site_age)),
        "n_states": len(sites),
        "python_quadrature": {
            **python, "sigma": fit["sigma"], "quadrature_nodes": fit["quadrature_nodes"],
            "profile_ci": (None if profile is None
                           else [float(np.exp(profile[0])), float(np.exp(profile[1]))]),
            "se_conditional_on_sigma": float(fit["se_conditional_on_sigma"][1]),
            "boundary_lrt": fit["boundary_lrt"], "singular": fit["singular"],
            "hessian": fit["hessian"], "max_absolute_gradient": gradient,
            "relative_gradient": abs(gradient / fit["log_likelihood"]),
            "variance_starts": fit["variance_starts"],
        },
        "lme4_reference": {
            "or": reference["or"], "ci": reference["ci"], "se": reference["se"],
            "source": str(REFERENCE.relative_to(RESULTS.parent)),
            "note": "fitted by glmer before lme4 began segfaulting at scale",
        },
        "log_or_absolute_difference": float(difference),
        "se_absolute_difference": float(abs(python["se"] - reference["se"])),
        "tolerance": TOLERANCE,
        "agrees": bool(difference < TOLERANCE),
    }


def design_comparison(groups, cells_per_group, nodes=REPORTED_NODES, rng=None):
    """Simulate a design, fit it both ways, and compare."""
    design, deaths, n, index, truth = simulate(rng, groups=groups, sigma=0.35,
                                               cells_per_group=cells_per_group)
    started = time.time()
    fit = glmm.fit_random_intercept(design, deaths, n, index, nodes=nodes)
    python_seconds = round(time.time() - started, 1)
    reference = fit_in_r(design, deaths, n, index, nodes)
    out = compare(fit, reference, truth)
    out.update({"n_cells": int(len(deaths)), "n_terms": int(design.shape[1]),
                "n_groups": groups, "true_sigma": 0.35,
                "python_seconds": python_seconds,
                "binomial_coefficient_constant": binomial_constant(deaths, n)})
    return out


def node_fit(nodes, rng):
    """One rung of the quadrature ladder, on the data every rung shares."""
    design, deaths, n, index, truth = simulate(unit_rng("node_ladder_data"), groups=32,
                                               sigma=0.35, cells_per_group=400)
    started = time.time()
    fit = glmm.fit_random_intercept(design, deaths, n, index, nodes=nodes)
    return {
        "nodes": nodes,
        "beta_1": float(fit["beta"][1]), "se_1": float(fit["se"][1]),
        "sigma": fit["sigma"], "log_likelihood": fit["log_likelihood"],
        "true_beta_1": float(truth[1]),
        "seconds": round(time.time() - started, 1),
    }


def coverage_batch(scenario, batch, rng):
    """One batch of replications: how often does each interval contain the truth?"""
    covered, profile_covered, boundary, profiled, unbracketed = 0, 0, 0, 0, 0
    for replication in range(COVERAGE_BATCH):
        design, deaths, n, index, truth = simulate(
            rng, groups=scenario["groups"], sigma=scenario["sigma"],
            terms=4, cells_per_group=60)
        fit = glmm.fit_random_intercept(design, deaths, n, index, nodes=11)
        boundary += int(fit["singular"])
        low = fit["beta"][1] - 1.96 * fit["se"][1]
        high = fit["beta"][1] + 1.96 * fit["se"][1]
        covered += int(low <= truth[1] <= high)
        if replication < PROFILE_PER_BATCH and not fit["singular"]:
            bounds = glmm.profile_interval(fit, 1)
            if bounds is None:
                unbracketed += 1
                continue
            profiled += 1
            profile_covered += int(bounds[0] <= truth[1] <= bounds[1])
    return {
        **scenario, "batch": batch, "replications": COVERAGE_BATCH,
        "wald_covered": covered, "profile_covered": profile_covered,
        "profiled": profiled, "profile_without_an_interval": unbracketed,
        "fits_at_the_boundary": boundary,
    }


def run_unit(name):
    """Compute one unit. Pure: the same name returns the same numbers."""
    rng = unit_rng(name)
    if name == "s9_against_lme4":
        return s9_against_lme4()
    if name == "reduced_design":
        return design_comparison(groups=12, cells_per_group=250, rng=rng)
    if name == "h6_shape":
        return design_comparison(groups=32, cells_per_group=5_000, rng=rng)
    if name.startswith("node_"):
        return node_fit(int(name.split("_")[1]), rng)
    if name.startswith("coverage_"):
        for scenario in COVERAGE_SCENARIOS:
            for batch in range(COVERAGE_REPLICATIONS // COVERAGE_BATCH):
                if coverage_name(scenario, batch) == name:
                    return coverage_batch(scenario, batch, rng)
    raise ValueError(f"no unit named {name}")


def wilson(successes, trials):
    """A binomial interval on a coverage estimate, so Monte Carlo error is visible."""
    if not trials:
        return None
    z, p = 1.96, successes / trials
    center = (p + z ** 2 / (2 * trials)) / (1 + z ** 2 / trials)
    half = (z * np.sqrt(p * (1 - p) / trials + z ** 2 / (4 * trials ** 2))
            / (1 + z ** 2 / trials))
    return [float(center - half), float(center + half)]


def assemble(payloads, started_at):
    """Turn the shards into the one file the paper and the reviewer read."""
    ladder = {int(name.split("_")[1]): payload for name, payload in payloads.items()
              if name.startswith("node_")}
    coverage = []
    for scenario in COVERAGE_SCENARIOS:
        batches = [payloads[coverage_name(scenario, batch)]
                   for batch in range(COVERAGE_REPLICATIONS // COVERAGE_BATCH)
                   if coverage_name(scenario, batch) in payloads]
        if not batches:
            continue
        total = sum(b["replications"] for b in batches)
        covered = sum(b["wald_covered"] for b in batches)
        profiled = sum(b["profiled"] for b in batches)
        profile_covered = sum(b["profile_covered"] for b in batches)
        boundary = sum(b["fits_at_the_boundary"] for b in batches)
        coverage.append({
            **scenario, "batches": len(batches), "replications": total,
            "wald_coverage": covered / total,
            "wald_coverage_interval": wilson(covered, total),
            "profile_coverage": (profile_covered / profiled) if profiled else None,
            "profile_coverage_interval": wilson(profile_covered, profiled),
            "profile_replications": profiled,
            "profile_without_an_interval": sum(
                b["profile_without_an_interval"] for b in batches),
            "fits_at_the_boundary": boundary,
            "profile_coverage_is_conditional_on_a_non_boundary_fit": bool(boundary),
        })

    output = {
        "timestamp": started_at,
        "run": run_metadata(f"glmm_validation_{started_at.replace(' ', 'T')}"),
        "estimator_fingerprint": fingerprint(),
        "units_completed": sorted(payloads),
        "units_expected": unit_names(),
    }
    for name in ("s9_against_lme4", "reduced_design", "h6_shape"):
        if name in payloads:
            output[name] = payloads[name]
    if ladder:
        finest = ladder[max(ladder)]
        output["quadrature_refinement"] = {
            "fits": {str(k): v for k, v in sorted(ladder.items())},
            "reported_nodes": REPORTED_NODES,
            "shift_to_the_finest_in_beta": (
                abs(ladder[REPORTED_NODES]["beta_1"] - finest["beta_1"])
                if REPORTED_NODES in ladder else None),
            "shift_to_the_finest_in_sigma": (
                abs(ladder[REPORTED_NODES]["sigma"] - finest["sigma"])
                if REPORTED_NODES in ladder else None),
        }
    if coverage:
        output["coverage"] = coverage
    return output


def shard_path(name, folder):
    return Path(folder) / f"{name}.json"


def load_shard(name, folder):
    """A shard counts only when it was produced by the estimator now on disk."""
    path = shard_path(name, folder)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return record["payload"] if record.get("fingerprint") == fingerprint() else None


def save_shard(name, payload, folder, seconds):
    path = shard_path(name, folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "unit": name, "fingerprint": fingerprint(), "seconds": round(seconds, 1),
        "finished_at": datetime.now().isoformat(timespec="seconds"), "payload": payload,
    }, indent=2))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shards", default=str(SHARDS), help="where unit shards are kept")
    parser.add_argument("--units", nargs="*", help="run only these units")
    parser.add_argument("--assemble-only", action="store_true",
                        help="write the result file from the shards already present")
    arguments = parser.parse_args()
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    folder = Path(arguments.shards)
    wanted = arguments.units or unit_names()

    payloads = {}
    for name in wanted:
        done = load_shard(name, folder)
        if done is not None:
            print(f"  [{datetime.now():%H:%M:%S}] {name}: already done", flush=True)
            payloads[name] = done
            continue
        if arguments.assemble_only:
            continue
        print(f"  [{datetime.now():%H:%M:%S}] {name}: running...", flush=True)
        started = time.time()
        payload = run_unit(name)
        save_shard(name, payload, folder, time.time() - started)
        payloads[name] = payload
        print(f"  [{datetime.now():%H:%M:%S}] {name}: {time.time() - started:.0f}s", flush=True)

    outpath = write_result(assemble(payloads, started_at), OUTPUT)
    print(f"\n  {len(payloads)} of {len(unit_names())} units, saved to {outpath}")


if __name__ == "__main__":
    main()
