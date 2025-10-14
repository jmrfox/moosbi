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
