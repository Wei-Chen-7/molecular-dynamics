"""Radial distribution function g(r): liquid structure and the ideal-gas limit."""

import numpy as np

from mdlj import (
    BerendsenThermostat,
    RadialDistribution,
    Simulation,
    System,
    initialize_positions,
    maxwell_boltzmann_velocities,
)


def _measure_gofr(rho, temperature, n=196, dim=2, seed=0, warmup=1200,
                  frames=120, stride=10, nbins=120):
    rng = np.random.default_rng(seed)
    pos, box = initialize_positions(n, density=rho, dim=dim, lattice="square")
    vel = maxwell_boltzmann_velocities(len(pos), dim, temperature, rng)
    system = System(pos, vel, box)
    thermo = BerendsenThermostat(temperature, tau=0.5)
    sim = Simulation(system, dt=0.005, thermostat=thermo)
    sim.run(warmup)  # equilibrate at the state point

    rdf = RadialDistribution(box, system.n, nbins=nbins)
    sim.run(frames * stride, sample_every=stride,
            callback=lambda s, _t: rdf.accumulate(s.positions))
    return rdf


def test_liquid_gofr_has_structure():
    """A dense LJ liquid shows excluded volume, a first peak and g -> 1."""
    rdf = _measure_gofr(rho=0.7, temperature=1.0, seed=1)
    r, g = rdf.result()

    # Excluded volume: g(r) ~ 0 inside the repulsive core.
    core = (r > 0.3) & (r < 0.9)
    assert np.all(g[core] < 0.1)

    # A pronounced first peak near the potential minimum (1.0 - 1.3 sigma).
    peak_region = r < 1.6
    peak_r = r[peak_region][np.argmax(g[peak_region])]
    assert 1.0 < peak_r < 1.3
    assert g.max() > 1.5  # genuine structure, not noise

    # Correct large-r normalization: the tail averages to 1.
    tail = r > 0.8 * r.max()
    assert abs(g[tail].mean() - 1.0) < 0.10


def test_low_density_gofr_is_flat():
    """A dilute (ideal-gas-like) system has g(r) ~ 1 everywhere beyond the core."""
    rdf = _measure_gofr(rho=0.05, temperature=2.0, n=120, seed=2,
                        warmup=600, frames=150, stride=8, nbins=40)
    r, g = rdf.result()
    # Beyond the interaction range g(r) should be flat at 1.
    beyond = r > 1.5
    assert abs(g[beyond].mean() - 1.0) < 0.10


def test_coordination_number_increases_to_bulk():
    """rho * integral of g(r) recovers a growing neighbour count."""
    rdf = _measure_gofr(rho=0.7, temperature=1.0, seed=4, frames=80)
    coordination = rdf.coordination_number()
    # Monotonic and reaching several neighbours within the first shells.
    assert coordination[-1] > coordination[0]
    # First coordination shell (out to the first g(r) minimum, ~1.5 sigma) has
    # several neighbours for a dense 2-D liquid.
    r = rdf.centers
    n_first_shell = coordination[np.searchsorted(r, 1.6)]
    assert 3.0 < n_first_shell < 8.0
