from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence
import numpy as np

from .simulator import Simulator
from .params import ParameterBank


def build_box_prior_from_pareto(
    bank: ParameterBank,
    pareto_matrix: np.ndarray,
    expand: float = 0.0,
):
    """Create an SBI BoxUniform prior from a Pareto decision matrix X.

    - pareto_matrix is shape (N, D) in the sampled-parameter space as returned by the MOO (e.g., res["X"]).
    - expand expands the bounds by a fraction of the range per dimension.
    - Bounds are clamped to the ParameterBank's [lower_bounds, upper_bounds].

    Returns a torch distribution (BoxUniform).
    """
    try:
        import torch
        from sbi.utils.torchutils import BoxUniform
    except Exception as e:
        raise ImportError("sbi (and torch) are required to build priors. Install sbi and torch.") from e

    X = np.asarray(pareto_matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of Pareto decision vectors.")
    if X.shape[1] != len(bank.sampled):
        raise ValueError(
            f"X has {X.shape[1]} columns but bank has {len(bank.sampled)} sampled parameters."
        )

    low = X.min(axis=0)
    high = X.max(axis=0)
    width = np.maximum(high - low, 1e-12)
    if expand and expand > 0:
        low = low - expand * width
        high = high + expand * width

    # Clamp to bank bounds
    low = np.maximum(low, np.asarray(bank.lower_bounds, dtype=float))
    high = np.minimum(high, np.asarray(bank.upper_bounds, dtype=float))

    low_t = torch.as_tensor(low, dtype=torch.float32)
    high_t = torch.as_tensor(high, dtype=torch.float32)
    return BoxUniform(low_t, high_t)


def make_sbi_simulator(
    simulator: Simulator,
    agg: str = "mean",
) -> Callable:
    """Wrap a Simulator into an SBI-compatible simulator that returns summary vectors.

    - Input: theta as torch.Tensor of shape (D,) or (N, D), with D=len(bank.sampled).
    - Output: torch.Tensor of shape (S,) or (N, S), where S is summary dimension.
    - Aggregates across seeds via 'mean' (default) or 'median'/'max'.
    """
    try:
        import torch
    except Exception as e:
        raise ImportError("torch is required for SBI simulation wrappers.") from e

    agg = agg.lower()

    def _aggregate(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            return arr
        if agg == "mean":
            return np.mean(arr, axis=0)
        if agg == "median":
            return np.median(arr, axis=0)
        if agg == "max":
            return np.max(arr, axis=0)
        return np.mean(arr, axis=0)

    def sim_fn(theta: "torch.Tensor") -> "torch.Tensor":  # quoted types to avoid hard torch dependency at import
        with torch.no_grad():
            theta_t = theta
            if theta_t.ndim == 1:
                theta_np = theta_t.cpu().numpy()
                params = simulator.to_instance(theta_np)
                summaries = []
                for seed in simulator.trials():
                    out = simulator.simulate(params, seed=seed)
                    s = simulator.summarize(out, params)
                    summaries.append(np.asarray(s, dtype=float))
                s_agg = _aggregate(np.stack(summaries, axis=0)) if len(summaries) > 1 else summaries[0]
                return torch.as_tensor(s_agg, dtype=torch.float32)
            elif theta_t.ndim == 2:
                outputs = []
                for row in theta_t.cpu().numpy():
                    params = simulator.to_instance(row)
                    summaries = []
                    for seed in simulator.trials():
                        out = simulator.simulate(params, seed=seed)
                        s = simulator.summarize(out, params)
                        summaries.append(np.asarray(s, dtype=float))
                    s_agg = _aggregate(np.stack(summaries, axis=0)) if len(summaries) > 1 else summaries[0]
                    outputs.append(s_agg)
                return torch.as_tensor(np.stack(outputs, axis=0), dtype=torch.float32)
            else:
                raise ValueError("theta must be 1D or 2D tensor")

    return sim_fn


def run_snpe(
    simulator: Simulator,
    prior,
    x_o: np.ndarray,
    num_simulations: int = 2000,
    agg: str = "mean",
    seed: Optional[int] = None,
):
    """Run SNPE with a Simulator and a prior, returning a posterior and artifacts.

    Parameters
    - simulator: Simulator instance with summary_fn defined.
    - prior: torch distribution (e.g., BoxUniform) over sampled parameters.
    - x_o: observed summary vector (numpy array, shape (S,)).
    - num_simulations: number of simulator calls to train SNPE.
    - agg: seed aggregation mode used by the SBI simulator wrapper.
    - seed: optional RNG seed forwarded to sbi.

    Returns dict with keys: posterior, inference, theta, x, x_o.
    """
    try:
        import torch
        from sbi.inference import SNPE
        from sbi.utils import prepare_for_sbi
        from sbi.utils.simulation import simulate_for_sbi
    except Exception as e:
        raise ImportError("sbi (and torch) are required for SNPE. Install sbi and torch.") from e

    sim_fn = make_sbi_simulator(simulator, agg=agg)
    sim_fn, prior = prepare_for_sbi(sim_fn, prior)

    theta, x = simulate_for_sbi(sim_fn, prior, num_simulations=num_simulations)

    inference = SNPE(prior=prior)
    density_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(density_estimator)

    x_o_t = torch.as_tensor(x_o, dtype=torch.float32)
    posterior.set_default_x(x_o_t)

    return {"posterior": posterior, "inference": inference, "theta": theta, "x": x, "x_o": x_o_t}
