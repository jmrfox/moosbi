from __future__ import annotations

from typing import Optional, Sequence, Tuple, List
import numpy as np

from .simulator import Simulator

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


def _aggregate(values: np.ndarray, mode: str = "mean", cvar_alpha: float = 0.2) -> float:
    if values.size == 0:
        return 0.0
    if mode == "mean":
        return float(np.mean(values))
    if mode == "median":
        return float(np.median(values))
    if mode == "max":
        return float(np.max(values))
    if mode == "cvar":
        k = max(1, int(np.ceil(cvar_alpha * values.size)))
        return float(np.mean(np.sort(values)[-k:]))
    return float(np.mean(values))


class SimulatorProblem(Problem):
    def __init__(
        self,
        simulator: Simulator,
        aggregate_mode: str = "mean",
        cvar_alpha: float = 0.2,
    ) -> None:
        self.sim = simulator
        self.aggregate_mode = aggregate_mode
        self.cvar_alpha = cvar_alpha

        n_var = len(self.sim.bank.sampled)
        n_obj = len(self.sim.objectives)
        n_constr = len(self.sim.constraints)
        xl = self.sim.bank.lower_bounds
        xu = self.sim.bank.upper_bounds

        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_constr, xl=xl, xu=xu, elementwise_evaluation=False)

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        n = X.shape[0]
        F = np.zeros((n, self.n_obj), dtype=float)
        G = np.zeros((n, self.n_ieq_constr), dtype=float) if self.n_ieq_constr > 0 else None

        for i in range(n):
            theta = X[i]
            params = self.sim.to_instance(theta)

            obj_trials: List[np.ndarray] = []
            con_trials: List[np.ndarray] = []

            for seed in self.sim.trials():
                out_i = self.sim.simulate(params, seed=seed)
                obj_trials.append(self.sim.objective_values(out_i, params))
                if self.n_ieq_constr > 0:
                    con_trials.append(self.sim.constraint_values(out_i, params))

            obj_trials_arr = np.vstack(obj_trials) if obj_trials else np.zeros((0, self.n_obj))
            for j in range(self.n_obj):
                F[i, j] = _aggregate(obj_trials_arr[:, j], mode=self.aggregate_mode, cvar_alpha=self.cvar_alpha)

            if self.n_ieq_constr > 0:
                con_trials_arr = np.vstack(con_trials) if con_trials else np.zeros((0, self.n_ieq_constr))
                G[i, :] = np.max(con_trials_arr, axis=0) if con_trials_arr.size else 0.0

        out["F"] = F
        if self.n_ieq_constr > 0 and G is not None:
            out["G"] = G


def optimize_with_pymoo(
    simulator: Simulator,
    pop_size: int = 64,
    n_gen: int = 100,
    seed: Optional[int] = None,
    aggregate_mode: str = "mean",
    cvar_alpha: float = 0.2,
    verbose: bool = True,
):
    problem = SimulatorProblem(simulator, aggregate_mode=aggregate_mode, cvar_alpha=cvar_alpha)
    algorithm = NSGA2(pop_size=pop_size)
    res = minimize(problem, algorithm, termination=("n_gen", n_gen), seed=seed, save_history=False, verbose=verbose)

    X = np.asarray(res.X, dtype=float)
    F = np.asarray(res.F, dtype=float)

    params_list = [simulator.to_instance(x) for x in X]

    return {"result": res, "X": X, "F": F, "params": params_list}
