"""Virial pressure and the ideal-gas limit."""

import numpy as np

from mdlj import (
    Simulation,
    System,
    analytic,
    initialize_positions,
    maxwell_boltzmann_velocities,
)
from mdlj.observables import virial_pressure


def test_virial_pressure_reduces_to_ideal_gas_at_low_density():
    """At low density the interaction term is tiny and P -> rho k_B T."""
    rho, temperature, dim = 0.01, 2.0, 2
    rng = np.random.default_rng(0)
    pos, box = initialize_positions(120, density=rho, dim=dim, lattice="square")
    vel = maxwell_boltzmann_velocities(len(pos), dim, temperature, rng)
    system = System(pos, vel, box)
    sim = Simulation(system, dt=0.005)
    sim.run(500)  # let it decorrelate a little

    pressures = []
    temps = []
    def sample(s, _t):
        temps.append(s.temperature)
        pressures.append(
            virial_pressure(s.temperature, s.density, s.virial, s.volume, s.dim)
        )
    sim.run(2000, sample_every=10, callback=sample)

    mean_p = float(np.mean(pressures))
    # Compare against the ideal-gas law at the measured mean temperature.
    ideal = analytic.ideal_gas_pressure(system.density, float(np.mean(temps)))
    assert abs(mean_p - ideal) / ideal < 0.05


def test_virial_pressure_positive_and_larger_when_repulsive():
    """A compressed (dense, hot) fluid has a clearly positive virial pressure."""
    rho, temperature, dim = 0.8, 2.0, 2
    rng = np.random.default_rng(1)
    pos, box = initialize_positions(144, density=rho, dim=dim, lattice="square")
    vel = maxwell_boltzmann_velocities(len(pos), dim, temperature, rng)
    system = System(pos, vel, box)
    sim = Simulation(system, dt=0.005)
    sim.run(500)
    pressures = []
    sim.run(1500, sample_every=10, callback=lambda s, _t: pressures.append(
        virial_pressure(s.temperature, s.density, s.virial, s.volume, s.dim)))
    # Dense repulsive fluid: pressure well above the ideal-gas value.
    assert np.mean(pressures) > system.density * temperature
