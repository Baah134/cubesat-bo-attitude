"""Shared matplotlib style: readable at IEEE two-column width (3.5 in)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL_W = 3.5
DBL_W = 7.16

METHOD_ORDER = ["bo", "tpe_iecon", "cmaes", "ga", "random", "pidtune", "analytic"]
LABELS = {"bo": "Constrained BO (GP)", "tpe_iecon": "TPE (IECON)", "cmaes": "CMA-ES",
          "ga": "GA", "random": "Random (Sobol)", "pidtune": "MATLAB pidtune",
          "analytic": "Analytic rule"}
COLORS = {"bo": "#1f4e9c", "tpe_iecon": "#4c9be8", "cmaes": "#d9822b", "ga": "#8e6c3a",
          "random": "#7f7f7f", "pidtune": "#2e8b57", "analytic": "#b22222"}


def setup():
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.2,
        "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150, "savefig.bbox": "tight",
        "font.family": "serif",
    })
    return plt


def save(fig, path):
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p.with_suffix(".png"), dpi=300)
    fig.savefig(p.with_suffix(".pdf"))
