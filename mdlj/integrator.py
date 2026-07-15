"""Velocity-Verlet integrator for microcanonical (NVE) dynamics.

The velocity-Verlet update advances positions and velocities as

    x(t + dt) = x(t) + v(t) dt + 1/2 a(t) dt^2
    a(t + dt) = F(x(t + dt)) / m
    v(t + dt) = v(t) + 1/2 [a(t) + a(t + dt)] dt

It is second-order accurate and time-reversible/symplectic.  As a result the
total energy has bounded fluctuations with no secular drift, and because the
force update is symmetric the total linear momentum is conserved to
floating-point precision.
"""

from __future__ import annotations

__all__ = ["velocity_verlet_step"]


def velocity_verlet_step(system, dt: float):
    """Advance ``system`` in place by one velocity-Verlet step.

    ``system`` must expose ``positions``, ``velocities``, ``masses``, ``forces``
    (the force at the current positions), a ``compute_forces`` method returning
    ``(forces, potential_energy, virial)`` and a ``wrap`` method for periodic
    boundaries.
    """
    inv_m = 1.0 / system.masses[:, None]
    accel = system.forces * inv_m

    # Drift positions using the current force, then re-wrap into the box.
    system.positions += system.velocities * dt + 0.5 * accel * dt * dt
    system.wrap()

    # New force at the updated positions.
    new_forces, potential_energy, virial = system.compute_forces()
    new_accel = new_forces * inv_m

    # Kick velocities with the average of the old and new accelerations.
    system.velocities += 0.5 * (accel + new_accel) * dt

    system.forces = new_forces
    system.potential_energy = potential_energy
    system.virial = virial
