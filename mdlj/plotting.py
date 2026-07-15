"""Headless-safe plotting helpers.

The matplotlib *Agg* backend is selected before ``pyplot`` is imported so that
figures render without a display (in CI, on a server, or over SSH).  The helpers
here produce the figures used by the example scripts and the README.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # must precede the pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = [
    "plot_energy",
    "plot_gofr",
    "plot_speed_distribution",
    "plot_positions",
]


def plot_energy(time, total, kinetic, potential, path=None):
    """Plot total, kinetic and potential energy against time."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(time, kinetic, lw=1.0, color="#d1495b", label="kinetic")
    ax.plot(time, potential, lw=1.0, color="#2e86ab", label="potential")
    ax.plot(time, total, lw=1.6, color="#1b1b1b", label="total")
    ax.set_xlabel(r"time  $t / \tau$")
    ax.set_ylabel(r"energy  $E / \epsilon$")
    ax.set_title("NVE energy conservation (velocity-Verlet)")
    ax.legend(loc="center right", frameon=False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140)
    return fig


def plot_gofr(r, g, path=None, label=None, extra=None, xmax=None):
    """Plot the radial distribution function ``g(r)``.

    ``extra`` is an optional list of ``(r, g, label)`` tuples for overlays
    (e.g. a low-density reference).  ``xmax`` caps the horizontal axis so the
    near-neighbour structure stays legible.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(r, g, lw=1.6, color="#2e86ab", label=label or r"$g(r)$")
    if extra:
        for rr, gg, lab in extra:
            ax.plot(rr, gg, lw=1.2, ls="--", color="#d1495b", label=lab)
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel(r"distance  $r / \sigma$")
    ax.set_ylabel(r"$g(r)$")
    ax.set_title("Radial distribution function")
    if xmax is not None:
        ax.set_xlim(0, xmax)
    ax.legend(frameon=False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140)
    return fig


def plot_speed_distribution(centers, density, analytic_v, analytic_f, path=None):
    """Overlay a measured speed histogram on the analytic Maxwell-Boltzmann curve."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    width = centers[1] - centers[0] if len(centers) > 1 else 0.1
    ax.bar(centers, density, width=width, color="#2e86ab", alpha=0.55,
           label="simulation")
    ax.plot(analytic_v, analytic_f, lw=1.8, color="#d1495b",
            label="Maxwell-Boltzmann")
    ax.set_xlabel(r"speed  $|v|$  $(\sigma / \tau)$")
    ax.set_ylabel("probability density")
    ax.set_title("Equilibrium speed distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140)
    return fig


def plot_positions(positions, box, path=None, ax=None):
    """Scatter plot of particle positions in the (2-D) box."""
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(5.0, 5.0))
    else:
        fig = ax.figure
    ax.scatter(positions[:, 0], positions[:, 1], s=60, color="#2e86ab",
               edgecolors="white", linewidths=0.5)
    ax.set_xlim(0, box[0])
    ax.set_ylim(0, box[1])
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x / \sigma$")
    ax.set_ylabel(r"$y / \sigma$")
    if created:
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=140)
    return fig, ax
