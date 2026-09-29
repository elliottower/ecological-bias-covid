"""
Binomial mixed model with one random intercept, by adaptive Gauss-Hermite quadrature.

`lme4::glmer` segfaults on this machine (R 4.1.2, lme4 1.1-33) on the cell table the
contextual model of H6 requires, so H6 is fitted here instead. The estimator is maximum
likelihood with the group integral evaluated by adaptive Gauss-Hermite quadrature, which
is what `glmer(..., nAGQ = k)` computes.

Fixed-effect uncertainty comes from the Hessian of that same marginal log-likelihood,
evaluated at two finite-difference step sizes and checked for symmetry and positive
definiteness, with a profile-likelihood interval available for one coefficient. The
variance component is also fitted at its boundary, exactly, by refitting the design as a
plain binomial GLM, so a solution at sigma = 0 is found rather than approached.

`validate_glmm.py` checks this code against `lme4` and against quadrature refinement and
writes the comparison to `results/glmm_validation.json`.
"""

import numpy as np
from scipy import optimize, stats
from scipy.special import expit, log_expit, roots_hermitenorm

BOUNDARY_TOLERANCE = 1e-6  # deviance gain over the sigma = 0 fit, below which the
                           # variance component sits at its boundary
HESSIAN_STEPS = (1e-4, 3e-4)
VARIANCE_STARTS = (0.1, 0.3, 1.0)  # the fit is run from each and the best is kept
# The optimizer is kept inside a range where the conditional mode exists: below this a
# variance component is indistinguishable from the boundary, which is fitted exactly and
# separately, and above it no state-level standard deviation on the log-odds scale is
# credible. Probing outside it produced mode searches with no solution to find.
SIGMA_BOUNDS = (1e-4, 10.0)
MODE_TOLERANCE = 1e-11


class ConvergenceError(RuntimeError):
    """The optimizer did not reach a solution."""


class RandomIntercept:
    """One binomial random-intercept model: its data, its quadrature rule, its likelihood.

    logit P(death) = X beta + b_group, b ~ N(0, sigma^2), on cells carrying `deaths` of
    `n`. The parameter vector is (beta, log sigma) throughout.
    """

    def __init__(self, design, deaths, n, groups, nodes=15):
        self.design = np.asarray(design, dtype=float)
        self.deaths = np.asarray(deaths, dtype=float)
        self.total = np.asarray(n, dtype=float)
        self.alive = self.total - self.deaths
        self.groups = np.asarray(groups, dtype=int)
        self.n_groups = int(self.groups.max()) + 1
        self.n_terms = self.design.shape[1]
        self.n_nodes = nodes
        self.quadrature_nodes, weights = roots_hermitenorm(nodes)
        self.quadrature_weights = weights / np.sqrt(2 * np.pi)
        self._mode = None
        self.last_mode_step = None
        self.failed_evaluations = 0

    def conditional_mode(self, offset, sigma, start=None, iterations=200):
        """Newton on u for each group, to tolerance: the integrand's mode and curvature.

        Every call iterates to `MODE_TOLERANCE`, so a warm start changes how many steps
        are taken and not the answer, and the likelihood is a function of its arguments
        alone rather than of the path the optimizer took to them.
        """
        u = np.zeros(self.n_groups) if start is None else start.copy()
        step = np.inf
        objective = self._penalized(offset, sigma, u)
        for _ in range(iterations):
            eta = offset + sigma * u[self.groups]
            probability = expit(eta)
            gradient = sigma * np.bincount(
                self.groups, weights=self.deaths - self.total * probability,
                minlength=self.n_groups) - u
            weight = self.total * probability * (1 - probability)
            curvature = -(sigma ** 2) * np.bincount(self.groups, weights=weight,
                                                    minlength=self.n_groups) - 1
            increment = gradient / curvature
            # Newton overshoots where the logistic is flat, so each group keeps halving
            # its own step until the penalized objective stops falling.
            scale = np.ones(self.n_groups)
            for _ in range(60):
                candidate = u - scale * increment
                candidate_objective = self._penalized(offset, sigma, candidate)
                improved = candidate_objective >= objective
                if improved.all():
                    break
                scale = np.where(improved, scale, 0.5 * scale)
            else:
                raise ConvergenceError(
                    f"the conditional-mode line search did not improve "
                    f"{int((~improved).sum())} of {self.n_groups} groups after 60 halvings")
            u, objective = candidate, candidate_objective
            step = float(np.max(np.abs(scale * increment)))
            if step < MODE_TOLERANCE:
                break
        else:
            raise ConvergenceError(f"conditional mode did not converge: step {step:.2e}")
        self.last_mode_step = step
        eta = offset + sigma * u[self.groups]
        probability = expit(eta)
        weight = self.total * probability * (1 - probability)
        curvature = (sigma ** 2) * np.bincount(self.groups, weights=weight,
                                               minlength=self.n_groups) + 1
        return u, curvature

    def _penalized(self, offset, sigma, u):
        """Each group's log integrand at u: its binomial term less the Gaussian penalty."""
        eta = offset + sigma * u[self.groups]
        per_cell = self.deaths * log_expit(eta) + self.alive * log_expit(-eta)
        return np.bincount(self.groups, weights=per_cell,
                           minlength=self.n_groups) - 0.5 * u ** 2

    def loglik(self, params, warm_start=True, anchor=None, strict=False):
        """The adaptive Gauss-Hermite marginal log-likelihood at (beta, log sigma).

        `anchor` fixes where the mode search starts. The per-group integrand is strictly
        concave with one maximum and the search runs to `MODE_TOLERANCE`, so the start
        changes how many Newton steps are taken and not the value; anchoring every
        evaluation of a Hessian at one point therefore keeps it both deterministic and
        far cheaper than restarting each search from zero.
        """
        beta, sigma = params[:-1], float(np.exp(params[-1]))
        offset = self.design @ beta
        start = anchor if anchor is not None else (self._mode if warm_start else None)
        try:
            mode, curvature = self.conditional_mode(offset, sigma, start=start)
        except ConvergenceError:
            if strict:
                raise
            # A point whose conditional mode cannot be found is reported to the optimizer
            # as infeasible and counted, rather than being silently accepted or aborting
            # a fit that a step away from it would have completed.
            self.failed_evaluations += 1
            self._mode = None
            return -np.inf
        self._mode = mode
        scale = 1.0 / np.sqrt(curvature)

        total = np.empty((self.n_groups, self.n_nodes))
        for k, (node, weight) in enumerate(zip(self.quadrature_nodes,
                                               self.quadrature_weights)):
            u = mode + scale * node
            eta = offset + sigma * u[self.groups]
            per_cell = self.deaths * log_expit(eta) + self.alive * log_expit(-eta)
            total[:, k] = (np.bincount(self.groups, weights=per_cell,
                                       minlength=self.n_groups)
                           - 0.5 * u ** 2 + 0.5 * node ** 2 + np.log(weight * scale))
        largest = total.max(axis=1, keepdims=True)
        return float(np.sum(np.log(np.sum(np.exp(total - largest), axis=1))
                            + largest.ravel()))

    def boundary_fit(self, start, iterations=100):
        """The sigma = 0 solution, exactly: the same design as a plain binomial GLM."""
        beta = np.asarray(start, dtype=float).copy()
        step = np.inf
        for _ in range(iterations):
            eta = self.design @ beta
            probability = expit(eta)
            weight = np.clip(self.total * probability * (1 - probability), 1e-12, None)
            working = eta + (self.deaths - self.total * probability) / weight
            weighted = self.design * weight[:, None]
            information = self.design.T @ weighted
            updated = np.linalg.solve(information, weighted.T @ working)
            step = float(np.max(np.abs(updated - beta)))
            beta = updated
            if step < 1e-11:
                break
        else:
            raise ConvergenceError(f"sigma = 0 fit did not converge: step {step:.2e}")
        eta = self.design @ beta
        loglik = float(np.sum(self.deaths * log_expit(eta) + self.alive * log_expit(-eta)))
        return beta, loglik, np.linalg.inv(information)

    def conditional_errors(self, beta, sigma):
        """Standard errors holding sigma fixed, from the joint information in (beta, b).

        The penalized information is [[X'WX, X'WZ], [Z'WX, Z'WZ + I/sigma^2]] at the
        conditional mode, and this is the beta block of its inverse. It is reported beside
        the marginal-likelihood errors as a computational check, never as the interval.
        """
        offset = self.design @ beta
        mode, _ = self.conditional_mode(offset, sigma)
        eta = offset + sigma * mode[self.groups]
        probability = expit(eta)
        weight = self.total * probability * (1 - probability)
        weighted = self.design * weight[:, None]
        information = self.design.T @ weighted
        cross = np.column_stack([
            np.bincount(self.groups, weights=weighted[:, j], minlength=self.n_groups)
            for j in range(self.n_terms)]).T
        group_information = (np.bincount(self.groups, weights=weight,
                                         minlength=self.n_groups) + 1.0 / sigma ** 2)
        schur = information - (cross / group_information) @ cross.T
        return np.sqrt(np.clip(np.diag(np.linalg.pinv(schur)), 0, None))


def _numerical_hessian(function, x, step):
    """Central second differences of a scalar function, evaluated cold at every point.

    The diagonal uses the three-point formula at the named step; the four-point mixed
    formula, applied to a diagonal entry, would instead be a central difference at twice
    the step, so writing it separately keeps every entry on the same step size.
    """
    n = len(x)
    center = function(x)
    hessian = np.zeros((n, n))
    for i in range(n):
        dx = np.zeros(n); dx[i] = step
        hessian[i, i] = (function(x + dx) - 2 * center + function(x - dx)) / step ** 2
        for j in range(i + 1, n):
            dy = np.zeros(n); dy[j] = step
            hessian[i, j] = hessian[j, i] = (
                function(x + dx + dy) - function(x + dx - dy)
                - function(x - dx + dy) + function(x - dx - dy)) / (4 * step ** 2)
    return hessian


def _marginal_covariance(model, params):
    """Fixed-effect covariance from the Hessian of the marginal log-likelihood.

    Returns the covariance at the first step size with the diagnostics a numerical Hessian
    has to pass before its intervals are used: symmetry, positive definiteness,
    conditioning, and agreement between two finite-difference step sizes.
    """
    model.loglik(params, warm_start=False)   # the solution's own mode, found cold once
    anchor = model._mode.copy()

    def negative(p):
        return -model.loglik(p, anchor=anchor)

    covariances, errors, symmetry, eigenvalues = [], [], [], []
    for step in HESSIAN_STEPS:
        hessian = _numerical_hessian(negative, params, step)
        symmetry.append(float(np.max(np.abs(hessian - hessian.T))))
        hessian = 0.5 * (hessian + hessian.T)
        eigenvalues.append(np.linalg.eigvalsh(hessian))
        covariances.append(np.linalg.inv(hessian))
        errors.append(np.sqrt(np.clip(np.diag(covariances[-1])[:-1], 0, None)))

    agreement = float(np.max(np.abs(errors[0] - errors[1])
                             / np.clip(errors[0], 1e-300, None)))
    return covariances[0], errors[0], {
        "steps": list(HESSIAN_STEPS),
        "max_relative_difference_between_steps": agreement,
        "asymmetry_before_symmetrization": symmetry[0],
        "positive_definite": bool(eigenvalues[0].min() > 0),
        "condition_number": float(eigenvalues[0].max() / eigenvalues[0].min()),
    }


def fit_random_intercept(design, deaths, n, groups, nodes=15, start=None, errors=True):
    """Fit logit P(death) = X beta + b_group by adaptive quadrature at `nodes` nodes.

    With `errors=False` the numerical Hessian is skipped and no standard errors are
    returned, which is what a reproduction check of another implementation's coefficients
    needs and costs a fraction of the time.

    Returns the fixed effects with their marginal-likelihood standard errors, sigma, the
    deviance difference against the exactly fitted sigma = 0 solution, the Hessian
    diagnostics and the optimizer's status. A solution at the boundary is reported as
    such, carrying the boundary fit's coefficients and errors.
    """
    model = RandomIntercept(design, deaths, n, groups, nodes=nodes)
    if start is None:
        proportion = np.clip((model.deaths + 0.5) / (model.total + 1.0), 1e-6, 1 - 1e-6)
        root = np.sqrt(model.total)
        start, *_ = np.linalg.lstsq(model.design * root[:, None],
                                    np.log(proportion / (1 - proportion)) * root,
                                    rcond=None)
    boundary_beta, boundary_loglik, boundary_covariance = model.boundary_fit(start)

    bounds = [(None, None)] * len(boundary_beta) + [
        (float(np.log(SIGMA_BOUNDS[0])), float(np.log(SIGMA_BOUNDS[1])))]
    attempts, failures = [], []
    for sigma_start in VARIANCE_STARTS:
        outcome = optimize.minimize(
            lambda p: -model.loglik(p),
            np.concatenate([boundary_beta, [np.log(sigma_start)]]),
            method="L-BFGS-B", bounds=bounds, options={"maxiter": 2000, "ftol": 1e-12})
        if outcome.success:
            attempts.append((model.loglik(outcome.x, warm_start=False, strict=True),
                             sigma_start, outcome))
        else:
            failures.append(f"sigma start {sigma_start}: {outcome.message}")
    if not attempts:
        raise ConvergenceError("; ".join(failures))

    interior_loglik, _, result = max(attempts, key=lambda entry: entry[0])
    starts = {
        "tried": list(VARIANCE_STARTS),
        "converged": [round(s, 4) for _, s, _ in attempts],
        "failed": failures,
        "largest_log_likelihood_spread": float(
            max(ll for ll, _, _ in attempts) - min(ll for ll, _, _ in attempts)),
        "largest_sigma_spread": float(
            max(np.exp(o.x[-1]) for _, _, o in attempts)
            - min(np.exp(o.x[-1]) for _, _, o in attempts)),
    }
    boundary_lrt = 2 * (interior_loglik - boundary_loglik)
    at_boundary = bool(boundary_lrt <= BOUNDARY_TOLERANCE)
    common = {
        "quadrature_nodes": nodes,
        "converged": True,
        "iterations": int(result.nit),
        "boundary_lrt": float(boundary_lrt),
        "conditional_mode_step_at_the_solution": model.last_mode_step,
        "optimizer_log_likelihood": float(-result.fun),
        "variance_starts": starts,
        "infeasible_evaluations": int(model.failed_evaluations),
        "sigma_bounds": list(SIGMA_BOUNDS),
        "boundary_p": float(0.5 * stats.chi2.sf(max(boundary_lrt, 0.0), 1)),
        "boundary_reference": ("50:50 mixture of a point mass at zero and chi-square on "
                               "one degree of freedom, the null for a variance component "
                               "at its boundary"),
        "singular": at_boundary,
        "model": model,
    }
    if at_boundary:
        return {
            **common,
            "beta": boundary_beta,
            "se": np.sqrt(np.clip(np.diag(boundary_covariance), 0, None)),
            "sigma": 0.0,
            "log_likelihood": boundary_loglik,
            "se_source": "the sigma = 0 fit; the variance component is at its boundary",
            "se_conditional_on_sigma": None,
            "hessian": None,
            "params": None,
        }

    sigma = float(np.exp(result.x[-1]))
    if not errors:
        return {
            **common,
            "beta": result.x[:-1],
            "se": None,
            "sigma": sigma,
            "log_likelihood": interior_loglik,
            "se_source": "not computed; this fit was asked for coefficients only",
            "se_conditional_on_sigma": None,
            "hessian": None,
            "params": result.x,
        }
    covariance, standard_errors, diagnostics = _marginal_covariance(model, result.x)
    return {
        **common,
        "beta": result.x[:-1],
        "se": standard_errors,
        "sigma": sigma,
        "log_likelihood": interior_loglik,
        "se_source": ("the Hessian of the adaptive-quadrature marginal log-likelihood in "
                      "(beta, log sigma); the beta block of its inverse"),
        "se_conditional_on_sigma": model.conditional_errors(result.x[:-1], sigma),
        "hessian": diagnostics,
        "params": result.x,
    }


def gradient_check(fit, step=1e-5):
    """The largest absolute finite-difference gradient of the log-likelihood at the fit."""
    if fit["params"] is None:
        return None
    model, params = fit["model"], fit["params"]
    gradient = []
    for i in range(len(params)):
        dx = np.zeros(len(params)); dx[i] = step
        gradient.append((model.loglik(params + dx, warm_start=False)
                         - model.loglik(params - dx, warm_start=False)) / (2 * step))
    return float(np.max(np.abs(gradient)))


def profile_interval(fit, index, level=0.95, maximum_width=8.0):
    """Profile-likelihood interval for one coefficient, inverting the likelihood ratio.

    The coefficient is held at a value, every other coefficient and log sigma are
    re-optimized, and the interval is where twice the log-likelihood deficit crosses the
    chi-square quantile. Preferred to a Wald interval when the groups are few.
    """
    if fit["params"] is None:
        return None
    model, params = fit["model"], fit["params"]
    target = float(stats.chi2.ppf(level, 1))
    point, error = float(params[index]), float(fit["se"][index])
    free = [i for i in range(len(params)) if i != index]

    def deficit(value):
        def negative(reduced):
            full = np.empty(len(params))
            full[index] = value
            full[free] = reduced
            return -model.loglik(full)
        outcome = optimize.minimize(negative, params[free], method="L-BFGS-B",
                                    options={"maxiter": 2000, "ftol": 1e-12})
        if not outcome.success:
            raise ConvergenceError(f"profile at {value:.6f}: {outcome.message}")
        return 2 * (fit["log_likelihood"] + outcome.fun) - target

    bounds = []
    for direction in (-1, 1):
        width = 2.0 * error
        try:
            while width <= maximum_width * error and deficit(point + direction * width) < 0:
                width *= 2
            if deficit(point + direction * width) < 0:
                return None  # the deficit never reached the cutoff inside the bracket
            low, high = sorted([point, point + direction * width])
            bounds.append(float(optimize.brentq(deficit, low, high, xtol=1e-6)))
        except (ConvergenceError, ValueError, RuntimeError):
            return None
    if not (np.all(np.isfinite(bounds)) and bounds[0] < point < bounds[1]):
        return None  # an interval is two finite endpoints around the estimate, or nothing
    return bounds


def wald(estimate, standard_error):
    return {
        "log_or": float(estimate),
        "se": float(standard_error),
        "or": float(np.exp(estimate)),
        "ci": [float(np.exp(estimate - 1.96 * standard_error)),
               float(np.exp(estimate + 1.96 * standard_error))],
        "p": float(2 * stats.norm.sf(abs(estimate / standard_error))),
    }
