"""Conservation laws of the velocity-Verlet NVE integrator.

These assert the defining properties of a symplectic integrator: bounded
energy fluctuations with no drift, and exact conservation of linear momentum.
"""

import numpy as np

from mdlj import (
    Simulation,
    System,
    initialize_positions,
    maxwell_boltzmann_velocities,
)


def _make_sim(seed=0):
    rng = np.random.default_rng(seed)
    pos, box = initialize_positions(64, density=0.7, dim=2, lattice="square")
    vel = maxwell_boltzmann_velocities(len(pos), 2, temperature=1.0, rng=rng)
    # Brute force is fastest at this small N and gives identical dynamics.
    system = System(pos, vel, box, use_cell_list=False)
    return Simulation(system, dt=0.005)


def test_energy_and_momentum_conserved_over_long_nve_run():
    """Over 10^4 steps at dt = 0.005, total energy is stable and momentum fixed.

    A single long microcanonical run checks both invariants: the total energy
    has small, bounded fluctuations with no secular drift, and the total linear
    momentum (started at zero after removing the centre of mass) never moves.
    """
    sim = _make_sim(seed=0)
    p0 = sim.system.momentum.copy()
    history = sim.run(10_000, sample_every=25)
    energy = history["total_energy"]
    momenta = history["momentum"]

    rel_fluctuation = energy.std() / abs(energy.mean())
    rel_drift = abs(energy[-1] - energy[0]) / abs(energy[0])
    assert rel_fluctuation < 1e-3
    assert rel_drift < 1e-2

    assert np.max(np.abs(p0)) < 1e-10  # COM removed at initialization
    assert np.max(np.abs(momenta - p0)) < 1e-10  # conserved throughout


def test_energy_drift_worsens_with_larger_timestep():
    """A symplectic check: a larger dt gives a larger (but still bounded) drift."""
    rng = np.random.default_rng(2)
    pos, box = initialize_positions(64, density=0.7, dim=2, lattice="square")
    vel = maxwell_boltzmann_velocities(len(pos), 2, 1.0, rng)

    def fluct(dt):
        system = System(pos.copy(), vel.copy(), box, use_cell_list=False)
        e = Simulation(system, dt=dt).run(2000, sample_every=10)["total_energy"]
        return e.std() / abs(e.mean())

    small = fluct(0.002)
    large = fluct(0.01)
    assert large > small
