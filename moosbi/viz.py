from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt

from .params import ParameterSet


def plot_pareto_scatter(
    F: np.ndarray,
    i: int = 0,
    j: int = 1,
    labels: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    s: float = 18.0,
    alpha: float = 0.9,
) -> plt.Axes:
    """Scatter plot of two objectives (i vs j) from an objective matrix F."""
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[1] < max(i, j) + 1:
        raise ValueError("F must be 2D with enough columns for indices i, j.")
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(F[:, i], F[:, j], s=s, alpha=alpha)
    if labels is None:
        labels = [f"f{k}" for k in range(F.shape[1])]
    ax.set_xlabel(labels[i])
    ax.set_ylabel(labels[j])
    ax.set_title(title or f"Pareto scatter: {labels[i]} vs {labels[j]}")
    return ax


def plot_pareto_pairs(
    F: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (8, 8),
    s: float = 12.0,
    alpha: float = 0.9,
    bins: int = 30,
) -> Tuple[plt.Figure, np.ndarray]:
    """Pairwise scatter grid of all objective columns in F."""
    F = np.asarray(F, dtype=float)
    n_obj = F.shape[1]
    if labels is None:
        labels = [f"f{k}" for k in range(n_obj)]
    fig, axes = plt.subplots(n_obj, n_obj, figsize=figsize)
    for i in range(n_obj):
        for j in range(n_obj):
            ax = axes[i, j] if n_obj > 1 else axes
            if i == j:
                ax.hist(F[:, i], bins=bins, color="C0", alpha=0.6)
                ax.set_ylabel("")
            else:
                ax.scatter(F[:, j], F[:, i], s=s, alpha=alpha)
            if i == n_obj - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel(labels[i])
            else:
                ax.set_ylabel("")
    fig.suptitle("Pairwise objectives", y=1.02)
    fig.tight_layout()
    return fig, axes


def _extract_param_array(
    params: Sequence[ParameterSet],
    name: str,
) -> np.ndarray:
    vals = [float(p[name]) for p in params]
    return np.asarray(vals, dtype=float)


def plot_param_hist(
    params: Sequence[ParameterSet],
    names: Sequence[str],
    truths: Optional[Sequence[Optional[float]]] = None,
    bins: int = 30,
    figsize: Tuple[float, float] = (4.0, 3.0),
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Histograms of selected parameters with optional ground-truth overlays."""
    if truths is None:
        truths = [None] * len(names)
    fig, axes = plt.subplots(1, len(names), figsize=(figsize[0] * len(names), figsize[1]))
    if len(names) == 1:
        axes = [axes]
    for ax, name, truth in zip(axes, names, truths):
        vals = _extract_param_array(params, name)
        ax.hist(vals, bins=bins, color="C0", alpha=0.6)
        if truth is not None:
            ax.axvline(float(truth), color="C3", linestyle="--", label="true")
            ax.legend(loc="best")
        ax.set_title(name)
        ax.set_xlabel(name)
        ax.set_ylabel("count")
    fig.tight_layout()
    return fig, axes  # type: ignore[return-value]


def plot_param_scatter(
    params: Sequence[ParameterSet],
    x_name: str,
    y_name: str,
    truths: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ax: Optional[plt.Axes] = None,
    s: float = 18.0,
    alpha: float = 0.9,
) -> plt.Axes:
    """Scatter of two parameters with optional true point overlay."""
    xs = _extract_param_array(params, x_name)
    ys = _extract_param_array(params, y_name)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(xs, ys, s=s, alpha=alpha)
    if truths is not None and truths[0] is not None and truths[1] is not None:
        ax.scatter([truths[0]], [truths[1]], c=["C3"], marker="x", s=80, label="true")
        ax.legend(loc="best")
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f"{x_name} vs {y_name}")
    ax.set_aspect("equal", adjustable="box")
    return ax
