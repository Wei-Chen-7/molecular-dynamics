"""Exercise the 3-D code path end to end (FCC lattice, NVE dynamics)."""

import numpy as np

from mdlj import (
    Simulation,
    System,
    initialize_positions,
    maxwell_boltzmann_velocities,
)


def test_fcc_initialisation_density():
    """The FCC initializer produces the requested 3-D number density."""
    pos, box = initialize_positions(256, density=0.8, dim=3, lattice="fcc")
    assert pos.shape[1] == 3
    assert abs(len(pos) / np.prod(box) - 0.8) < 1e-9


def test_3d_nve_conserves_energy_and_momentum():
    """A short 3-D NVE run conserves energy and momentum."""
    rng = np.random.default_rng(0)
    pos, box = initialize_positions(108, density=0.6, dim=3, lattice="fcc")
    vel = maxwell_boltzmann_velocities(len(pos), 3, temperature=1.2, rng=rng)
    system = System(pos, vel, box)
    sim = Simulation(system, dt=0.004)
    p0 = system.momentum.copy()
    hist = sim.run(1500, sample_every=20)
    e = hist["total_energy"]
    assert e.std() / abs(e.mean()) < 1e-2
    assert np.max(np.abs(hist["momentum"] - p0)) < 1e-10


def test_3d_equipartition_variance():
    """3-D per-component velocity variance equals kT after equilibration."""
    target = 1.2
    rng = np.random.default_rng(1)
    pos, box = initialize_positions(108, density=0.6, dim=3, lattice="fcc")
    vel = maxwell_boltzmann_velocities(len(pos), 3, target, rng)
    system = System(pos, vel, box)
    sim = Simulation(system, dt=0.004)
    sim.run(1000)
    comps = []
    temps = []
    sim.run(1600, sample_every=20, callback=lambda s, _t: (
        comps.append(s.velocities.copy()), temps.append(s.temperature)))
    v = np.concatenate(comps, axis=0)
    # In NVE the mean temperature settles near (but not exactly at) the target;
    # equipartition requires equal variance across the three components, each
    # equal to the measured temperature (kT/m).
    var = v.var(axis=0)
    assert np.allclose(var, var.mean(), rtol=0.05)
    assert np.allclose(var, float(np.mean(temps)), rtol=0.05)
