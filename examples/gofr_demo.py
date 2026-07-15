"""Radial distribution function g(r) demo.

Equilibrates a two-dimensional Lennard-Jones liquid at a standard state point
(rho* = 0.7, T* = 1.0) and accumulates the radial distribution function g(r)
over many frames.  It also runs a dilute (ideal-gas-like) system whose g(r) is
flat at 1 for comparison.  The figure is written to ``figures/gofr.png`` and the
first-peak location, peak height, tail average and coordination number are
printed.

Run with:  python examples/gofr_demo.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdlj import (  # noqa: E402
    BerendsenThermostat,
    RadialDistribution,
    Simulation,
    System,
    initialize_positions,
    maxwell_boltzmann_velocities,
)
from mdlj.plotting import plot_gofr  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "figures")


def measure_gofr(n, density, temperature, seed, warmup=4000, frames=400,
                 stride=5, nbins=150):
    rng = np.random.default_rng(seed)
    pos, box = initialize_positions(n, density, dim=2, lattice="square")
    vel = maxwell_boltzmann_velocities(len(pos), 2, temperature, rng)
    system = System(pos, vel, box)
    sim = Simulation(system, dt=0.005, thermostat=BerendsenThermostat(temperature, 0.5))
    sim.run(warmup)  # equilibrate at the state point
    rdf = RadialDistribution(box, system.n, nbins=nbins)
    sim.run(frames * stride, sample_every=stride,
            callback=lambda s, _t: rdf.accumulate(s.positions))
    return rdf


def main():
    seed = 2026
    print("Radial distribution function demo (2-D Lennard-Jones)")
    print(f"  seed={seed}")

    # Dense liquid at the standard state point.
    print("  liquid : N=400  rho*=0.7  T*=1.0")
    liquid = measure_gofr(400, 0.7, 1.0, seed)
    r, g = liquid.result()

    # Dilute, ideal-gas-like reference.
    print("  dilute : N=200  rho*=0.05 T*=2.0")
    dilute = measure_gofr(200, 0.05, 2.0, seed + 1, warmup=2000,
                          frames=500, stride=4, nbins=60)
    r_d, g_d = dilute.result()

    peak_i = np.argmax(g[r < 1.6])
    peak_r, peak_h = r[r < 1.6][peak_i], g[r < 1.6][peak_i]
    tail = r > 0.8 * r.max()
    coordination = liquid.coordination_number()
    n_first_shell = coordination[np.searchsorted(r, 1.6)]

    print("\nResults (liquid)")
    print(f"  first-peak location : r = {peak_r:.3f} sigma")
    print(f"  first-peak height   : g = {peak_h:.3f}")
    print(f"  tail average g(r)   : {g[tail].mean():.3f}   (should be ~1)")
    print(f"  neighbours in first shell (r<1.6) : {n_first_shell:.2f}")
    print(f"  dilute tail average g(r) : {g_d[r_d > 1.5].mean():.3f}   (should be ~1)")

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "gofr.png")
    plot_gofr(r, g, path=out, label=r"liquid ($\rho^*=0.7,\ T^*=1.0$)",
              extra=[(r_d, g_d, r"dilute ($\rho^*=0.05$)")], xmax=8.0)
    print(f"\nsaved figure -> {out}")


if __name__ == "__main__":
    main()
