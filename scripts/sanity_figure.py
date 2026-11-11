"""Sanity figure: time histories for fixed designs (analytic + pidtune) on the
slew and hold scenarios. Usage: python -m scripts.sanity_figure --stage A"""
import argparse

import numpy as np

from cubesat.params import DEG
from cubesat.problem import Problem
from optim.analytic import analytic_gains
from optim.pidtune import pidtune_gains
from scripts.style import COLORS, DBL_W, LABELS, save, setup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="A")
    a = ap.parse_args()
    plt = setup()
    p = Problem(stage=a.stage)
    fig, ax = plt.subplots(4, 2, figsize=(DBL_W, 6.0), sharex="col")
    for name, g in [("analytic", analytic_gains(p)), ("pidtune", pidtune_gains(p))]:
        _, (rs, rh) = p.simulate(g)
        for col, r in enumerate((rs, rh)):
            err = np.atleast_2d(r.err.T).T if r.err.ndim == 1 else r.err
            ax[0, col].plot(r.t, np.abs(r.err) / DEG if r.err.ndim == 1 else r.err / DEG,
                            color=COLORS[name], label=LABELS[name])
            ax[1, col].plot(r.t, np.atleast_2d(r.rate.T).T / DEG, color=COLORS[name], lw=0.8)
            tc = np.arange(len(r.u_cmd)) * p.sim.ctrl_dt
            ax[2, col].plot(tc, np.atleast_2d(r.u_cmd.T).T * 1e3, color=COLORS[name], lw=0.8)
            ax[3, col].plot(r.t, np.atleast_2d(r.h.T).T * 1e3, color=COLORS[name], lw=0.8)
    for col, title in enumerate(["Slew (30°)", "Pointing hold (0.5° offset)"]):
        ax[0, col].set_title(title)
        ax[0, col].set_yscale("log")
        ax[0, col].axhline(0.1, color="k", ls=":", lw=0.8)
        ax[2, col].axhline(p.sc.tau_max * 1e3, color="k", ls=":", lw=0.8)
        ax[2, col].axhline(-p.sc.tau_max * 1e3, color="k", ls=":", lw=0.8)
        ax[3, col].axhline(p.sc.h_max * 1e3, color="k", ls=":", lw=0.8)
        ax[3, col].axhline(-p.sc.h_max * 1e3, color="k", ls=":", lw=0.8)
        ax[3, col].set_xlabel("time (s)")
    ax[0, 0].set_ylabel("|pointing error| (deg)")
    ax[1, 0].set_ylabel("body rate (deg/s)")
    ax[2, 0].set_ylabel("torque (mN m)")
    ax[3, 0].set_ylabel("wheel h (mN m s)")
    ax[0, 0].legend(loc="upper right")
    fig.align_ylabels(ax[:, 0])
    save(fig, f"figures/sanity_stage{a.stage}")


if __name__ == "__main__":
    main()
