"""Thermostats for canonical (NVT) sampling.

A thermostat couples the system to a heat bath at a target temperature.  Two
schemes are provided:

* **Berendsen** — a weak, deterministic velocity rescaling that relaxes the
  temperature toward the target with a time constant ``tau_T``.  It is simple
  and stable for driving a system to a set point, though it does not sample the
  exact canonical ensemble.
* **Andersen** — stochastic collisions with the bath: at each step a random
  subset of particles has its velocity redrawn from the Maxwell-Boltzmann
  distribution.  This does sample the canonical ensemble.

Both expose ``apply(system, dt)`` and are used by
:class:`mdlj.simulation.Simulation`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["BerendsenThermostat", "AndersenThermostat"]


class BerendsenThermostat:
    """Berendsen weak-coupling thermostat.

    After each step the velocities are scaled by

        lambda = sqrt(1 + (dt / tau_T) * (T_target / T - 1)),

    which nudges the instantaneous temperature ``T`` toward ``T_target`` with a
    relaxation time ``tau_T``.  Larger ``tau_T`` means gentler coupling.

    Parameters
    ----------
    target_temperature : float
        Set-point temperature ``T_target`` (units of ``epsilon / k_B``).
    tau : float
        Coupling time constant ``tau_T`` (units of ``tau``).  Must be
        ``>= dt``; a common choice is ``tau_T ~ 100 * dt``.
    """

    def __init__(self, target_temperature: float, tau: float = 0.5):
        self.target_temperature = float(target_temperature)
        self.tau = float(tau)

    def apply(self, system, dt: float):
        t = system.temperature
        if t <= 0.0:
            return
        ratio = self.target_temperature / t
        lam = np.sqrt(1.0 + (dt / self.tau) * (ratio - 1.0))
        system.velocities *= lam


class AndersenThermostat:
    """Andersen stochastic-collision thermostat (canonical ensemble).

    Each particle independently "collides" with the bath at rate ``nu``: with
    probability ``nu * dt`` per step its velocity is replaced by a fresh draw
    from the Maxwell-Boltzmann distribution at the target temperature (each
    component Gaussian with variance ``k_B T / m``).

    Parameters
    ----------
    target_temperature : float
        Bath temperature.
    collision_rate : float
        Collision frequency ``nu`` (units of ``1 / tau``).
    rng : numpy.random.Generator, optional
        Seeded generator for reproducibility.
    """

    def __init__(self, target_temperature: float, collision_rate: float = 1.0,
                 rng=None):
        self.target_temperature = float(target_temperature)
        self.collision_rate = float(collision_rate)
        self.rng = rng if rng is not None else np.random.default_rng()

    def apply(self, system, dt: float):
        prob = self.collision_rate * dt
        hit = self.rng.random(system.n) < prob
        if not np.any(hit):
            return
        std = np.sqrt(self.target_temperature / system.masses[hit])[:, None]
        system.velocities[hit] = self.rng.normal(
            0.0, 1.0, size=(int(hit.sum()), system.dim)
        ) * std
