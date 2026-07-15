"""Maxwell-Boltzmann speed distribution demo.

Equilibrates a two-dimensional Lennard-Jones fluid at T* = 1.0 and histograms
the particle speeds sampled over many frames, overlaying the closed-form
Maxwell-Boltzmann (Rayleigh, in 2-D) speed distribution.  Measured and analytic
mean / most-probable / rms speeds are printed together with a Kolmogorov-Smirnov
test against the analytic curve.  The figure is written to
``figures/maxwell_boltzmann.png``.

Run with:  python examples/maxwell_boltzmann_demo.py
"""

import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdlj import (  # noqa: E402
    BerendsenThermostat,
    Simulation,
    System,
    analytic,
    initialize_positions,
    maxwell_boltzmann_velocities,
)
from mdlj.plotting import plot_speed_distribution  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "figures")


def main():
    seed = 2026
    n, density, temperature, dim = 400, 0.7, 1.0, 2
    print("Maxwell-Boltzmann speed distribution demo (2-D Lennard-Jones)")
    print(f"  seed={seed}  N={n}  rho*={density}  T*={temperature}")

    rng = np.random.default_rng(seed)
    pos, box = initialize_positions(n, density, dim=dim, lattice="square")
    vel = maxwell_boltzmann_velocities(len(pos), dim, temperature, rng)
    system = System(pos, vel, box)
    sim = Simulation(system, dt=0.005,
                     thermostat=BerendsenThermostat(temperature, 0.5))
    sim.run(3000)  # equilibrate

    speeds = []
    for _ in range(60):  # sample well-separated frames
        sim.run(50)
        speeds.append(np.linalg.norm(sim.system.velocities, axis=1))
    speeds = np.concatenate(speeds)

    # Speed histogram (probability density) for the plot and the mode.
    counts, edges = np.histogram(speeds, bins=45, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    mean_meas = speeds.mean()
    vp_meas = centers[np.argmax(counts)]  # most-probable speed = histogram mode
    rms_meas = np.sqrt(np.mean(speeds ** 2))
    stat, pvalue = stats.kstest(
        speeds, lambda x: analytic.maxwell_boltzmann_speed_cdf(x, temperature, dim))

    print("\nResults                measured   analytic")
    print(f"  mean speed <v>        {mean_meas:8.4f}  {analytic.mean_speed(temperature, dim):9.4f}")
    print(f"  most-probable v_p     {vp_meas:8.4f}  {analytic.most_probable_speed(temperature, dim):9.4f}")
    print(f"  rms speed v_rms       {rms_meas:8.4f}  {analytic.rms_speed(temperature, dim):9.4f}")
    print(f"  KS statistic = {stat:.4f}   p-value = {pvalue:.3f}")

    density_hist = counts
    vv = np.linspace(0, centers.max() * 1.1, 300)
    ff = analytic.maxwell_boltzmann_speed_pdf(vv, temperature, dim)

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "maxwell_boltzmann.png")
    plot_speed_distribution(centers, density_hist, vv, ff, path=out)
    print(f"\nsaved figure -> {out}")


if __name__ == "__main__":
    main()
