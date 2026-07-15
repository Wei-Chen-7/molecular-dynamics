"""Physical observables: energies, temperature, g(r), speeds and pressure.

These are standalone, array-based helpers (useful in tests and analysis) that
mirror the convenience properties on :class:`mdlj.simulation.System`.  All
quantities are in reduced Lennard-Jones units.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "kinetic_energy",
    "temperature",
    "speed_histogram",
    "virial_pressure",
    "minimum_image_distances",
    "RadialDistribution",
]


def kinetic_energy(velocities, masses=1.0) -> float:
    """Total kinetic energy ``0.5 * sum(m v^2)``."""
    velocities = np.asarray(velocities, dtype=float)
    masses = np.broadcast_to(np.asarray(masses, dtype=float), (velocities.shape[0],))
    return 0.5 * float(np.sum(masses[:, None] * velocities ** 2))


def temperature(velocities, masses=1.0, remove_com: bool = True) -> float:
    """Instantaneous temperature ``2 KE / (N_dof k_B)``.

    With ``remove_com`` (the default) the centre-of-mass degrees of freedom are
    excluded: ``N_dof = d N - d``.  This matches the conserved-momentum
    constraint of the dynamics.
    """
    velocities = np.asarray(velocities, dtype=float)
    n, dim = velocities.shape
    ke = kinetic_energy(velocities, masses)
    n_dof = dim * n - (dim if remove_com else 0)
    if n_dof <= 0:
        return 0.0
    return 2.0 * ke / n_dof


def speed_histogram(velocities, bins: int = 40, vmax=None):
    """Normalized histogram of particle speeds ``|v|``.

    Returns
    -------
    centers : ndarray
        Bin centres.
    density : ndarray
        Probability density (integrates to 1).
    """
    speeds = np.linalg.norm(np.asarray(velocities, dtype=float), axis=1)
    if vmax is None:
        vmax = speeds.max() * 1.05
    counts, edges = np.histogram(speeds, bins=bins, range=(0.0, vmax), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def virial_pressure(temperature_value: float, density: float, virial: float,
                    volume: float, dim: int) -> float:
    """Instantaneous pressure from the virial theorem.

        P = rho k_B T + (1 / (d V)) * sum_{i<j} r_ij . F_ij

    The first term is the ideal-gas (kinetic) contribution and the second the
    interaction (virial) contribution.  In the dilute limit the virial term
    vanishes and ``P -> rho k_B T``.
    """
    return density * temperature_value + virial / (dim * volume)


def minimum_image_distances(positions, box):
    """All unique pair distances under the minimum image convention.

    Returns a 1-D array of the ``N (N-1) / 2`` pair distances.  Intended for
    small/medium ``N`` (analysis and tests), not the force hot path.
    """
    positions = np.asarray(positions, dtype=float)
    box = np.atleast_1d(np.asarray(box, dtype=float))
    n = positions.shape[0]
    i, j = np.triu_indices(n, k=1)
    rij = positions[i] - positions[j]
    rij -= box * np.round(rij / box)
    return np.linalg.norm(rij, axis=1)


class RadialDistribution:
    """Accumulate the radial distribution function ``g(r)`` over many frames.

    ``g(r)`` measures the density of particles at distance ``r`` from a
    reference particle, relative to the ideal-gas expectation ``rho``.  For each
    frame the minimum-image pair distances are histogrammed; the counts are then
    normalized by the number of pairs an ideal gas of the same density would put
    in each spherical/circular shell, so that ``g(r) -> 1`` at large ``r``.

    Parameters
    ----------
    box : array_like
        Box side length(s).
    n_particles : int
        Number of particles (assumed constant across frames).
    nbins : int
        Number of radial bins.
    rmax : float, optional
        Maximum radius; defaults to half the smallest box side (the limit of
        validity of the minimum image convention).
    """

    def __init__(self, box, n_particles: int, nbins: int = 100, rmax=None):
        self.box = np.atleast_1d(np.asarray(box, dtype=float))
        self.dim = self.box.size
        self.n = n_particles
        self.volume = float(np.prod(self.box))
        self.density = self.n / self.volume
        if rmax is None:
            rmax = 0.5 * float(self.box.min())
        self.rmax = float(rmax)
        self.nbins = nbins
        self.edges = np.linspace(0.0, self.rmax, nbins + 1)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.counts = np.zeros(nbins)
        self.nframes = 0

    def accumulate(self, positions):
        """Add one frame's worth of pair distances to the histogram."""
        distances = minimum_image_distances(positions, self.box)
        hist, _ = np.histogram(distances, bins=self.edges)
        self.counts += hist
        self.nframes += 1

    def _shell_volumes(self):
        """Volume (2-D area) of each radial shell."""
        if self.dim == 2:
            return np.pi * (self.edges[1:] ** 2 - self.edges[:-1] ** 2)
        return (4.0 / 3.0) * np.pi * (self.edges[1:] ** 3 - self.edges[:-1] ** 3)

    def result(self):
        """Return ``(centers, g)`` averaged over all accumulated frames.

        Each unordered pair contributes once, so the ideal-gas reference count
        per shell is ``0.5 * N * rho * shell_volume``.
        """
        if self.nframes == 0:
            raise RuntimeError("no frames accumulated")
        shell = self._shell_volumes()
        ideal = 0.5 * self.n * self.density * shell
        g = self.counts / (self.nframes * ideal)
        return self.centers, g

    def coordination_number(self):
        """Cumulative coordination number ``rho * integral of g(r) dV``.

        Running integral of the number of neighbours out to each radius.  Over
        the whole accessible range it approaches ``N - 1``.
        """
        _, g = self.result()
        shell = self._shell_volumes()
        return np.cumsum(self.density * g * shell)
