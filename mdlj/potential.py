"""Lennard-Jones 12-6 pair potential and force.

All quantities are expressed in *reduced Lennard-Jones units* with
``epsilon = sigma = m = k_B = 1``.  In these units lengths are measured in
sigma, energies in epsilon, and time in ``tau = sigma * sqrt(m / epsilon)``.

The (unshifted) pair potential is

    U(r) = 4 * epsilon * [ (sigma / r)^12 - (sigma / r)^6 ],

which has a single minimum at ``r_min = 2**(1/6) * sigma`` (about
``1.122462 * sigma``) where ``U(r_min) = -epsilon`` and the force is exactly
zero.  The pair force is the negative gradient of the potential; its magnitude
is

    F(r) = (24 * epsilon / r) * [ 2 * (sigma / r)^12 - (sigma / r)^6 ].

For a simulation the potential is truncated at a cutoff ``r_c`` (default
``2.5 * sigma``) and *shifted* so that the energy is continuous (equal to zero)
at the cutoff:

    U_shift(r) = U(r) - U(r_c)   for r < r_c,   and   0   for r >= r_c.

The unshifted well depth at the cutoff, ``U(2.5 * sigma)``, is about
``-0.0163 * epsilon``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["lj_potential", "lj_force_magnitude", "LennardJones"]


def lj_potential(r, epsilon: float = 1.0, sigma: float = 1.0):
    """Unshifted Lennard-Jones potential ``U(r)``.

    Parameters
    ----------
    r : float or ndarray
        Pair separation(s), in units of sigma.
    epsilon, sigma : float
        Energy and length scale of the potential (both 1 in reduced units).

    Returns
    -------
    float or ndarray
        ``4 * epsilon * [(sigma/r)**12 - (sigma/r)**6]``.
    """
    sr6 = (sigma / np.asarray(r, dtype=float)) ** 6
    return 4.0 * epsilon * (sr6 * sr6 - sr6)


def lj_force_magnitude(r, epsilon: float = 1.0, sigma: float = 1.0):
    """Magnitude of the Lennard-Jones pair force ``F(r) = -dU/dr``.

    Positive values are repulsive (pointing to increase ``r``), negative values
    attractive.  The force vanishes at ``r_min = 2**(1/6) * sigma``.

    Parameters
    ----------
    r : float or ndarray
        Pair separation(s), in units of sigma.
    epsilon, sigma : float
        Energy and length scale of the potential.
    """
    r = np.asarray(r, dtype=float)
    sr6 = (sigma / r) ** 6
    sr12 = sr6 * sr6
    return 24.0 * epsilon / r * (2.0 * sr12 - sr6)


class LennardJones:
    """Truncated-and-shifted Lennard-Jones interaction.

    The potential is evaluated up to a cutoff ``rc`` and shifted by ``U(rc)`` so
    the energy goes continuously to zero at the cutoff.  This class stores the
    parameters and precomputes the squared cutoff and the energy shift used by
    the vectorized force routine in :mod:`mdlj.simulation`.

    Parameters
    ----------
    epsilon, sigma : float
        Energy and length scale (both 1 in reduced units).
    rc : float
        Cutoff radius, in units of sigma (default ``2.5``).
    """

    def __init__(self, epsilon: float = 1.0, sigma: float = 1.0, rc: float = 2.5):
        self.epsilon = float(epsilon)
        self.sigma = float(sigma)
        self.rc = float(rc)
        self.rc2 = self.rc * self.rc
        # Energy offset that makes U_shift(rc) == 0 exactly.
        self.shift = float(lj_potential(self.rc, self.epsilon, self.sigma))

    def energy(self, r):
        """Shifted pair energy ``U_shift(r)`` (zero at and beyond the cutoff)."""
        r = np.asarray(r, dtype=float)
        u = lj_potential(r, self.epsilon, self.sigma) - self.shift
        return np.where(r < self.rc, u, 0.0)

    def force_magnitude(self, r):
        """Shifted pair force magnitude (zero at and beyond the cutoff).

        The shift is a constant, so it does not change the force inside the
        cutoff; the force is simply set to zero for ``r >= rc``.
        """
        r = np.asarray(r, dtype=float)
        f = lj_force_magnitude(r, self.epsilon, self.sigma)
        return np.where(r < self.rc, f, 0.0)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"LennardJones(epsilon={self.epsilon}, sigma={self.sigma}, "
            f"rc={self.rc})"
        )
