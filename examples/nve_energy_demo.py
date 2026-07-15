"""NVE energy conservation demo.

Runs a microcanonical (constant energy) Lennard-Jones simulation and plots the
total, kinetic and potential energy against time.  The total energy stays flat
while kinetic and potential energy continuously exchange as the initial lattice
melts into a liquid.  The measured energy drift and fluctuation are printed and
the figure is written to ``figures/energy_conservation.png`` (embedded in the
README).

Run with:  python examples/nve_energy_demo.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdlj import (  # noqa: E402
    Simulation,
    System,
    initialize_positions,
    maxwell_boltzmann_velocities,
)
from mdlj.plotting import plot_energy  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "figures")


def main():
    seed = 2026
    n, density, temperature, dim = 144, 0.7, 1.0, 2
    dt, nsteps = 0.005, 20_000

    print("NVE energy conservation demo (reduced Lennard-Jones units)")
    print(f"  seed={seed}  N={n}  rho={density}  T0={temperature}  "
          f"dt={dt}  steps={nsteps}  dim={dim}")

    rng = np.random.default_rng(seed)
    pos, box = initialize_positions(n, density, dim=dim, lattice="square")
    vel = maxwell_boltzmann_velocities(n, dim, temperature, rng)
    system = System(pos, vel, box)
    sim = Simulation(system, dt=dt)  # no thermostat -> NVE

    history = sim.run(nsteps, sample_every=10)
    energy = history["total_energy"]

    rel_fluctuation = energy.std() / abs(energy.mean())
    rel_drift = abs(energy[-1] - energy[0]) / abs(energy[0])
    max_dp = np.max(np.abs(history["momentum"] - history["momentum"][0]))

    print("\nResults")
    print(f"  mean total energy per particle : {energy.mean() / n:+.5f} eps")
    print(f"  relative energy fluctuation    : {rel_fluctuation:.2e}"
          "   (std(E)/|<E>|)")
    print(f"  relative energy drift          : {rel_drift:.2e}"
          "   (|E_f - E_i|/|E_i|)")
    print(f"  max |P(t) - P(0)|              : {max_dp:.2e}"
          "   (linear momentum)")

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "energy_conservation.png")
    plot_energy(history["time"], energy, history["kinetic"],
                history["potential"], path=out)
    print(f"\nsaved figure -> {out}")


if __name__ == "__main__":
    main()
