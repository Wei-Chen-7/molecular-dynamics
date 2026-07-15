"""Equilibrium statistical mechanics: equipartition, speeds, thermostatting."""

import numpy as np
from scipy import stats

from mdlj import (
    BerendsenThermostat,
    Simulation,
    System,
    analytic,
    initialize_positions,
    maxwell_boltzmann_velocities,
)


def _equilibrated_system(n=144, rho=0.7, dim=2, temperature=1.0, seed=0,
                         thermostat=None, warmup=1000):
    rng = np.random.default_rng(seed)
    pos, box = initialize_positions(n, density=rho, dim=dim, lattice="square")
    vel = maxwell_boltzmann_velocities(len(pos), dim, temperature, rng)
    system = System(pos, vel, box)
    sim = Simulation(system, dt=0.005, thermostat=thermostat)
    sim.run(warmup)
    return sim


def test_equipartition_temperature_and_variance():
    """NVT run: time-averaged T hits target and KE is shared equally."""
    target = 1.0
    thermo = BerendsenThermostat(target, tau=0.5)
    sim = _equilibrated_system(temperature=target, thermostat=thermo, seed=1)

    temps = []
    comp_sq = []  # per-component v^2, pooled over frames
    def sample(system, _t):
        temps.append(system.temperature)
        comp_sq.append(system.velocities.copy())
    sim.run(1800, sample_every=20, callback=sample)

    mean_t = float(np.mean(temps))
    assert abs(mean_t - target) / target < 0.05

    v = np.concatenate(comp_sq, axis=0)  # (samples, dim)
    var_per_component = v.var(axis=0)    # should each equal kT/m = target
    assert np.allclose(var_per_component, target, rtol=0.05)


def test_maxwell_boltzmann_speed_distribution():
    """Equilibrium speeds follow Maxwell-Boltzmann (moments + KS test)."""
    target, dim = 1.0, 2
    thermo = BerendsenThermostat(target, tau=0.5)
    sim = _equilibrated_system(temperature=target, dim=dim, seed=2, warmup=800,
                               thermostat=thermo)

    speeds = []
    # Sample well-separated frames to keep the KS samples near-independent.
    for _ in range(24):
        sim.run(80)
        speeds.append(np.linalg.norm(sim.system.velocities, axis=1))
    speeds = np.concatenate(speeds)

    mean_measured = speeds.mean()
    rms_measured = np.sqrt(np.mean(speeds ** 2))
    assert abs(mean_measured - analytic.mean_speed(target, dim)) < 0.05 * analytic.mean_speed(target, dim)
    assert abs(rms_measured - analytic.rms_speed(target, dim)) < 0.05 * analytic.rms_speed(target, dim)

    # Kolmogorov-Smirnov against the closed-form speed CDF.
    stat, pvalue = stats.kstest(
        speeds, lambda x: analytic.maxwell_boltzmann_speed_cdf(x, target, dim)
    )
    assert stat < 0.05 or pvalue > 0.05


def test_velocity_components_are_gaussian():
    """Each velocity component is Gaussian with variance kT (KS test)."""
    target, dim = 1.0, 2
    thermo = BerendsenThermostat(target, tau=0.5)
    sim = _equilibrated_system(temperature=target, dim=dim, seed=7, warmup=800,
                               thermostat=thermo)
    comps = []
    for _ in range(24):
        sim.run(80)
        comps.append(sim.system.velocities.ravel())
    comps = np.concatenate(comps)
    stat, pvalue = stats.kstest(comps, "norm", args=(0.0, np.sqrt(target)))
    assert stat < 0.05 or pvalue > 0.05


def test_berendsen_thermostat_drives_to_target():
    """Starting far from the target, the thermostat brings T to the set point."""
    target = 1.5
    thermo = BerendsenThermostat(target, tau=0.5)
    # Initialize hot (T = 3.0) and cool toward 1.5.
    sim = _equilibrated_system(temperature=3.0, thermostat=thermo, seed=3,
                               warmup=1200)
    temps = sim.run(1500, sample_every=20)["temperature"]
    assert abs(temps.mean() - target) / target < 0.05
