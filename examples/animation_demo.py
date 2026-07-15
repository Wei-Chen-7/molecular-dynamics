"""Particle animation demo.

Renders the moving Lennard-Jones particles as a scatter animation inside the
periodic box and saves it as a GIF using matplotlib's Pillow writer, so it works
headlessly (no display required).  The GIF is written to
``figures/animation.gif``.

Run with:  python examples/animation_demo.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdlj import (  # noqa: E402
    BerendsenThermostat,
    Simulation,
    System,
    initialize_positions,
    maxwell_boltzmann_velocities,
)
from mdlj.plotting import plt  # noqa: E402  (Agg backend already selected)
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "figures")


def main():
    seed = 2026
    n, density, temperature = 100, 0.5, 1.0
    n_frames, steps_per_frame = 150, 15

    print("Particle animation demo (2-D Lennard-Jones)")
    print(f"  seed={seed}  N={n}  rho*={density}  T*={temperature}  "
          f"frames={n_frames}")

    rng = np.random.default_rng(seed)
    pos, box = initialize_positions(n, density, dim=2, lattice="square")
    vel = maxwell_boltzmann_velocities(len(pos), 2, temperature, rng)
    system = System(pos, vel, box)
    sim = Simulation(system, dt=0.005,
                     thermostat=BerendsenThermostat(temperature, 1.0))
    sim.run(500)  # equilibrate before recording

    # Pre-compute the trajectory so rendering is decoupled from the physics.
    frames = []
    for _ in range(n_frames):
        sim.run(steps_per_frame)
        frames.append(system.positions.copy())

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    scat = ax.scatter(frames[0][:, 0], frames[0][:, 1], s=80, color="#2e86ab",
                      edgecolors="white", linewidths=0.5)
    ax.set_xlim(0, box[0])
    ax.set_ylim(0, box[1])
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x / \sigma$")
    ax.set_ylabel(r"$y / \sigma$")
    ax.set_title(r"Lennard-Jones fluid ($T^* = 1.0$)")

    def update(i):
        scat.set_offsets(frames[i])
        return (scat,)

    anim = FuncAnimation(fig, update, frames=n_frames, blit=True)

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "animation.gif")
    anim.save(out, writer=PillowWriter(fps=20))
    plt.close(fig)
    print(f"\nsaved animation -> {out}")


if __name__ == "__main__":
    main()
