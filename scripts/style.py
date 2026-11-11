"""Shared matplotlib style: readable at IEEE two-column width (3.5 in)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL_W = 3.5
DBL_W = 7.16

METHOD_ORDER = ["bo_robust", "bo", "wie_rule", "tpe_iecon", "cmaes", "ga", "random_search", "pidtune", "analytic"]
LABELS = {"bo_robust": "Robust BO (worst of 8 plants)",
          "bo": "Constrained BO (GP)", "tpe_iecon": "TPE (IECON)", "cmaes": "CMA-ES",
          "ga": "GA", "random_search": "Random (Sobol)", "pidtune": "MATLAB pidtune",
          "analytic": "Analytic rule", "wie_rule": "Expert law, rule-designed (Wie-Lu)"}
COLORS = {"bo_robust": "#6a3d9a", "bo": "#1f4e9c", "tpe_iecon": "#4c9be8", "cmaes": "#d9822b", "ga": "#8e6c3a",
          "random_search": "#7f7f7f", "pidtune": "#2e8b57", "analytic": "#b22222", "wie_rule": "#c51b7d", "bo_wie": "#e7298a"}


def label(stage, method):
    """Display label; stage W is the expert (Wie-Lu) control law instead of PID."""
    if stage == "W" and method == "bo":
        return "Expert law, BO-tuned (Wie-Lu)"
    if stage == "W" and method == "wie_rule":
        return LABELS["wie_rule"]
    return LABELS.get(method, method) + (" [PID]" if stage == "B" else "")


def color(stage, method):
    return COLORS["bo_wie"] if (stage == "W" and method == "bo") else COLORS.get(method, "k")


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
