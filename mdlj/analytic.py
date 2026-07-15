"""Closed-form reference results used to validate the simulation.

These functions provide the analytic Maxwell-Boltzmann velocity/speed
distributions, the equipartition energy and ideal-gas relations that the
measured observables are checked against in the tests.  Everything is in
reduced Lennard-Jones units (``epsilon = sigma = m = k_B = 1``); the ``mass``
and ``kB`` arguments are kept explicit for clarity.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

__all__ = [
    "maxwell_boltzmann_component_pdf",
    "maxwell_boltzmann_speed_pdf",
    "maxwell_boltzmann_speed_cdf",
    "mean_speed",
    "most_probable_speed",
    "rms_speed",
    "equipartition_kinetic_energy",
    "ideal_gas_pressure",
]


def maxwell_boltzmann_component_pdf(vx, temperature, mass=1.0, kB=1.0):
    """PDF of a single velocity component: Gaussian with variance ``kB T / m``."""
    var = kB * temperature / mass
    return np.exp(-np.asarray(vx, float) ** 2 / (2.0 * var)) / np.sqrt(2.0 * np.pi * var)


def maxwell_boltzmann_speed_pdf(v, temperature, dim, mass=1.0, kB=1.0):
    """Maxwell-Boltzmann *speed* distribution ``f(v)`` in ``dim`` dimensions.

    * 2-D (Rayleigh): ``f(v) = (m v / kT) exp(-m v^2 / (2 kT))``.
    * 3-D (Maxwell):   ``f(v) = 4 pi v^2 (m / (2 pi kT))^{3/2} exp(-m v^2 / (2 kT))``.
    """
    v = np.asarray(v, dtype=float)
    a2 = kB * temperature / mass  # variance of each component = kT/m
    if dim == 2:
        return (v / a2) * np.exp(-v * v / (2.0 * a2))
    if dim == 3:
        return (4.0 * np.pi * v * v * (1.0 / (2.0 * np.pi * a2)) ** 1.5
                * np.exp(-v * v / (2.0 * a2)))
    raise ValueError("dim must be 2 or 3")


def maxwell_boltzmann_speed_cdf(v, temperature, dim, mass=1.0, kB=1.0):
    """CDF of the Maxwell-Boltzmann speed distribution (for KS tests)."""
    v = np.asarray(v, dtype=float)
    a = np.sqrt(kB * temperature / mass)  # component standard deviation
    if dim == 2:
        return 1.0 - np.exp(-v * v / (2.0 * a * a))
    if dim == 3:
        x = v / a
        return erf(x / np.sqrt(2.0)) - np.sqrt(2.0 / np.pi) * x * np.exp(-x * x / 2.0)
    raise ValueError("dim must be 2 or 3")


def mean_speed(temperature, dim, mass=1.0, kB=1.0):
    """Mean speed ``<v>``: ``sqrt(pi kT / 2m)`` (2-D), ``sqrt(8 kT / pi m)`` (3-D)."""
    a2 = kB * temperature / mass
    if dim == 2:
        return np.sqrt(np.pi * a2 / 2.0)
    if dim == 3:
        return np.sqrt(8.0 * a2 / np.pi)
    raise ValueError("dim must be 2 or 3")


def most_probable_speed(temperature, dim, mass=1.0, kB=1.0):
    """Most probable speed ``v_p``: ``sqrt(kT/m)`` (2-D), ``sqrt(2kT/m)`` (3-D)."""
    a2 = kB * temperature / mass
    if dim == 2:
        return np.sqrt(a2)
    if dim == 3:
        return np.sqrt(2.0 * a2)
    raise ValueError("dim must be 2 or 3")


def rms_speed(temperature, dim, mass=1.0, kB=1.0):
    """Root-mean-square speed ``v_rms = sqrt(d kT / m)``."""
    return np.sqrt(dim * kB * temperature / mass)


def equipartition_kinetic_energy(n, dim, temperature, kB=1.0, remove_com=True):
    """Expected total kinetic energy ``(1/2) N_dof k_B T`` from equipartition.

    With ``remove_com`` the centre-of-mass motion is excluded:
    ``N_dof = d N - d``.
    """
    n_dof = dim * n - (dim if remove_com else 0)
    return 0.5 * n_dof * kB * temperature


def ideal_gas_pressure(density, temperature, kB=1.0):
    """Ideal-gas pressure ``P = rho k_B T``."""
    return density * kB * temperature
