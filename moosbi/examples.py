from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence
import numpy as np

from .scip import ParameterSet, ParameterBank
from .simulator import Simulator


def _lag1_autocorr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 0.0
    x0 = x[:-1]
    x1 = x[1:]
    v = np.var(x, ddof=0)
    if v <= 0.0:
        return 0.0
    c = np.mean((x0 - np.mean(x0)) * (x1 - np.mean(x1)))
    return float(c / v)


class OUExampleSimulator(Simulator):
    """Ornstein–Uhlenbeck (OU) example simulator.

    Model
    -----
    The OU process is a mean-reverting Gaussian diffusion defined by the SDE

        dX_t = θ (μ − X_t) dt + σ dW_t,

    where θ > 0 is the mean-reversion rate, μ is the long-run mean, σ > 0 is the
    diffusion scale, and W_t is a standard Wiener process. For small dt, the
    discrete-time lag-1 autocorrelation is approximately exp(−θ·dt).

    Inputs as parameters
    --------------------
    In moosbi, all model inputs (including simulation controls and targets) are
    represented as entries of a `ParameterSet`. This simulator expects the
    following keys (all provided via the `ParameterSet`):

    - "theta" (float, sampled): mean-reversion rate (> 0)
    - "mu" (float, sampled): long-run mean
    - "sigma" (float, sampled): diffusion scale (> 0)
    - "T" (float, fixed): total simulated time (default 1.0)
    - "dt" (float, fixed): time step for Euler–Maruyama (default 1e-3)
    - "burn_in" (int, fixed): number of initial steps to drop (default 0)
    - "x0" (float, fixed): initial state (default μ)

    Optionally, targets for objectives may also be included as fixed parameters:
    - "target_mean", "target_var", "target_ac1".

    Outputs
    -------
    The simulator returns a dict containing:
    - "series": np.ndarray of the simulated trajectory after burn-in
    - "mean": float, empirical mean of the series
    - "var": float, empirical variance of the series (ddof=0)
    - "ac1": float, empirical lag-1 autocorrelation of the series

    Numerical scheme and randomness
    -------------------------------
    Integration uses Euler–Maruyama:

        X_{t+dt} = X_t + θ (μ − X_t) dt + σ sqrt(dt) ξ,   ξ ~ N(0, 1)

    If a seed is supplied to `simulate(params, seed)`, `numpy.random.seed(seed)`
    is set before generating the trajectory to provide reproducibility.

    Usage
    -----
    Construct a `ParameterBank` with sampled model parameters and fixed inputs/targets,
    draw a `theta` vector if `theta_sampling=True`, convert to a `ParameterSet`, and call
    `simulate(params, seed)`. Objective and summary functions should consume the
    returned output dict together with the same `ParameterSet` used for simulation.
    """
    def simulate(self, params: ParameterSet, seed: Optional[int] = None) -> Any:
        theta = float(params["theta"])  # mean reversion rate > 0
        mu = float(params["mu"])       # long-run mean
        sigma = float(params["sigma"])  # volatility > 0

        # All simulation inputs are parameters (fixed or sampled) in params
        T = float(params.get("T", 1.0))
        dt = float(params.get("dt", 1e-3))
        burn_in = int(params.get("burn_in", 0))
        x0 = float(params.get("x0", mu))

        n_total = max(1, int(np.ceil(T / dt))) + burn_in
        if seed is not None:
            np.random.seed(seed)

        x = np.empty(n_total, dtype=float)
        x[0] = x0
        sqrt_dt = np.sqrt(dt)
        for t in range(1, n_total):
            dW = np.random.normal(0.0, 1.0)
            x[t] = x[t - 1] + theta * (mu - x[t - 1]) * dt + sigma * sqrt_dt * dW

        if burn_in > 0:
            series = x[burn_in:]
        else:
            series = x

        out = {
            "series": series,
            "mean": float(np.mean(series)),
            "var": float(np.var(series, ddof=0)),
            "ac1": _lag1_autocorr(series),
        }
        return out


def ou_obj_mean(output: Any, params: ParameterSet) -> float:
    target = float(params.get("target_mean", 0.0))
    diff = float(output["mean"]) - target
    return diff * diff


def ou_obj_var(output: Any, params: ParameterSet) -> float:
    target = float(params.get("target_var", 1.0))
    diff = float(output["var"]) - target
    return diff * diff


def ou_obj_ac1(output: Any, params: ParameterSet) -> float:
    target = float(params.get("target_ac1", 0.0))
    diff = float(output["ac1"]) - target
    return diff * diff


def ou_summary(output: Any, params: ParameterSet) -> np.ndarray:
    return np.asarray([output["mean"], output["var"], output["ac1"]], dtype=float)


def make_ou_example(
    bank: ParameterBank,
    seeds: Optional[Sequence[int]] = None,
    use_objectives: Sequence[Callable[[Any, ParameterSet], float]] = (ou_obj_mean, ou_obj_var, ou_obj_ac1),
    summary_fn: Optional[Callable[[Any, ParameterSet], np.ndarray]] = ou_summary,
) -> OUExampleSimulator:
    return OUExampleSimulator(
        parameter_bank=bank,
        objectives=list(use_objectives),
        constraints=None,
        seeds=list(seeds) if seeds is not None else None,
        summary_fn=summary_fn,
    )
