from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, List
import numpy as np

from .scip import ParameterBank, ParameterSet


class Simulator(ABC):
    def __init__(
        self,
        parameter_bank: ParameterBank,
        objectives: Sequence[Callable[[Any, ParameterSet], float]],
        constraints: Optional[Sequence[Callable[[Any, ParameterSet], float]]] = None,
        seeds: Optional[Sequence[int]] = None,
        summary_fn: Optional[Callable[[Any, ParameterSet], np.ndarray]] = None,
    ) -> None:
        self.bank = parameter_bank
        self.objectives = list(objectives)
        self.constraints = list(constraints) if constraints is not None else []
        self.seeds = list(seeds) if seeds is not None else [None]
        self.summary_fn = summary_fn

    @abstractmethod
    def simulate(self, params: ParameterSet, seed: Optional[int] = None) -> Any:
        pass

    def to_instance(self, theta: np.ndarray) -> ParameterSet:
        theta = np.asarray(theta, dtype=float)
        return self.bank.theta_to_instance(theta)

    def objective_values(self, output: Any, params: ParameterSet) -> np.ndarray:
        vals: List[float] = [float(fn(output, params)) for fn in self.objectives]
        return np.asarray(vals, dtype=float)

    def constraint_values(self, output: Any, params: ParameterSet) -> np.ndarray:
        if not self.constraints:
            return np.zeros(0, dtype=float)
        vals: List[float] = [float(fn(output, params)) for fn in self.constraints]
        return np.asarray(vals, dtype=float)

    def summarize(self, output: Any, params: ParameterSet) -> np.ndarray:
        if self.summary_fn is None:
            raise RuntimeError("summary_fn is required for summarization")
        s = self.summary_fn(output, params)
        return np.asarray(s, dtype=float)

    def trials(self) -> Iterable[Optional[int]]:
        for seed in self.seeds:
            yield seed
