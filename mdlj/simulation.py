"""System state, periodic boundaries and the NVE/NVT run loop.

A :class:`System` holds the positions, velocities and masses of the particles
together with the box and the interaction, and knows how to compute the total
force, potential energy and virial.  Pair separations use the *minimum image
convention*: for a box of side ``L`` each component of a displacement is mapped
into ``[-L/2, L/2]`` by ``dx -> dx - L * round(dx / L)``.  Forces obey Newton's
third law (``F_ij = -F_ji``) so the total force on the system is zero.

A :class:`Simulation` wraps a system with an integrator (velocity-Verlet) and
an optional thermostat and provides ``step`` / ``run``.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .integrator import velocity_verlet_step
from .neighbors import CellList, all_pairs
from .potential import LennardJones

__all__ = ["pair_forces", "System", "Simulation"]


def pair_forces(positions, box, potential: LennardJones, i_idx, j_idx):
    """Forces, potential energy and virial from a list of candidate pairs.

    This is the single vectorized hot path shared by the brute-force and
    cell-list code paths, which is why they give bitwise-comparable results.
    Displacements use the minimum image convention and are filtered by the
    cutoff; contributions are scattered back onto the two partners of each pair
    with opposite sign (Newton's third law).

    Parameters
    ----------
    positions : ndarray, shape (N, d)
        Particle positions.
    box : ndarray, shape (d,)
        Box side lengths.
    potential : LennardJones
        The truncated-and-shifted interaction.
    i_idx, j_idx : ndarray
        Index arrays describing candidate pairs (each unordered pair once).

    Returns
    -------
    forces : ndarray, shape (N, d)
        Total force on each particle.
    potential_energy : float
        Total shifted potential energy.
    virial : float
        ``sum_{i<j} r_ij . F_ij``, used for the virial pressure.
    """
    positions = np.asarray(positions, dtype=float)
    box = np.asarray(box, dtype=float)
    n, dim = positions.shape
    forces = np.zeros((n, dim))
    i_idx = np.asarray(i_idx)
    j_idx = np.asarray(j_idx)
    if i_idx.size == 0:
        return forces, 0.0, 0.0

    rij = positions[i_idx] - positions[j_idx]
    rij -= box * np.round(rij / box)  # minimum image
    r2 = np.einsum("ij,ij->i", rij, rij)

    within = r2 < potential.rc2
    if not np.any(within):
        return forces, 0.0, 0.0
    rij = rij[within]
    r2 = r2[within]
    i_idx = i_idx[within]
    j_idx = j_idx[within]

    inv_r2 = 1.0 / r2
    sr6 = (potential.sigma * potential.sigma * inv_r2) ** 3
    sr12 = sr6 * sr6

    # Shifted pair energies.
    e_pair = 4.0 * potential.epsilon * (sr12 - sr6) - potential.shift
    potential_energy = float(e_pair.sum())

    # F_ij = 24 eps (2 sr12 - sr6) * r_ij / r^2 = factor * r_ij.
    factor = 24.0 * potential.epsilon * (2.0 * sr12 - sr6) * inv_r2
    fvec = factor[:, None] * rij

    # Scatter with opposite signs; bincount is far faster than np.add.at.
    for k in range(dim):
        forces[:, k] += np.bincount(i_idx, weights=fvec[:, k], minlength=n)
        forces[:, k] -= np.bincount(j_idx, weights=fvec[:, k], minlength=n)

    virial = float(np.sum(factor * r2))  # sum r_ij . F_ij
    return forces, potential_energy, virial


class System:
    """Container for particle state, the box and the interaction.

    Parameters
    ----------
    positions : ndarray, shape (N, d)
        Initial positions (wrapped into the box on construction).
    velocities : ndarray, shape (N, d)
        Initial velocities.
    box : array_like
        Box side length(s): a scalar (square/cubic) or one value per dimension.
    potential : LennardJones, optional
        Interaction (defaults to the standard ``rc = 2.5`` LJ).
    masses : ndarray, optional
        Per-particle masses (default all 1).
    use_cell_list : bool or "auto"
        Whether to use the linked-cell list.  ``"auto"`` (the default) uses it
        only when it is valid (>= 3 cells per dimension) *and* the system is
        large enough (``N >= 120``) for the ``O(N)`` scaling to beat the
        vectorized brute-force ``O(N**2)`` loop; below that the brute-force list
        is faster.  Either way the results are identical.
    """

    _CELL_LIST_MIN_N = 120

    def __init__(self, positions, velocities, box, potential=None, masses=None,
                 use_cell_list="auto"):
        self.positions = np.array(positions, dtype=float)
        self.velocities = np.array(velocities, dtype=float)
        self.n, self.dim = self.positions.shape
        self.box = np.broadcast_to(
            np.atleast_1d(np.asarray(box, dtype=float)), (self.dim,)
        ).copy()
        self.potential = potential if potential is not None else LennardJones()
        if masses is None:
            self.masses = np.ones(self.n)
        else:
            self.masses = np.broadcast_to(
                np.asarray(masses, dtype=float), (self.n,)
            ).copy()
        self._cell_list = CellList(self.box, self.potential.rc)
        if use_cell_list == "auto":
            self.use_cell_list = (self._cell_list.usable
                                  and self.n >= self._CELL_LIST_MIN_N)
        else:
            self.use_cell_list = bool(use_cell_list)

        self.wrap()
        self.forces, self.potential_energy, self.virial = self.compute_forces()

    # -- geometry -----------------------------------------------------------
    def wrap(self):
        """Wrap all positions back into ``[0, L)`` (positions are periodic)."""
        self.positions %= self.box

    @property
    def volume(self) -> float:
        return float(np.prod(self.box))

    @property
    def density(self) -> float:
        """Number density ``rho = N / V``."""
        return self.n / self.volume

    # -- forces -------------------------------------------------------------
    def compute_forces(self):
        """Total forces, potential energy and virial for the current state."""
        if self.use_cell_list and self._cell_list.usable:
            i_idx, j_idx = self._cell_list.candidate_pairs(self.positions)
        else:
            i_idx, j_idx = all_pairs(self.n)
        return pair_forces(self.positions, self.box, self.potential, i_idx, j_idx)

    # -- observables --------------------------------------------------------
    @property
    def kinetic_energy(self) -> float:
        """Total kinetic energy ``0.5 * sum(m v^2)``."""
        return 0.5 * float(np.sum(self.masses[:, None] * self.velocities ** 2))

    @property
    def total_energy(self) -> float:
        return self.kinetic_energy + self.potential_energy

    @property
    def temperature(self) -> float:
        """Instantaneous temperature from equipartition, ``2 KE / (N_dof k_B)``.

        The number of degrees of freedom is ``d N - d``; the ``-d`` removes the
        conserved centre-of-mass motion.
        """
        n_dof = self.dim * self.n - self.dim
        if n_dof <= 0:
            return 0.0
        return 2.0 * self.kinetic_energy / n_dof

    @property
    def momentum(self) -> np.ndarray:
        """Total linear momentum ``sum(m v)`` (a d-vector)."""
        return np.sum(self.masses[:, None] * self.velocities, axis=0)

    def remove_center_of_mass_velocity(self):
        """Subtract the centre-of-mass velocity so net momentum is zero."""
        v_cm = np.sum(self.masses[:, None] * self.velocities, axis=0) / self.masses.sum()
        self.velocities -= v_cm


class Simulation:
    """Drive a :class:`System` with velocity-Verlet and an optional thermostat.

    Parameters
    ----------
    system : System
        The system to integrate.
    dt : float
        Time step, in units of ``tau`` (default ``0.005``).
    thermostat : object, optional
        Anything with an ``apply(system, dt)`` method (see
        :mod:`mdlj.thermostat`).  ``None`` gives pure NVE dynamics.
    """

    def __init__(self, system: System, dt: float = 0.005, thermostat=None):
        self.system = system
        self.dt = float(dt)
        self.thermostat = thermostat
        self.time = 0.0
        self.step_count = 0

    def step(self):
        """Advance the system by one velocity-Verlet step (+ thermostat)."""
        velocity_verlet_step(self.system, self.dt)
        if self.thermostat is not None:
            self.thermostat.apply(self.system, self.dt)
        self.time += self.dt
        self.step_count += 1

    def run(self, nsteps: int, sample_every: int = 1, callback=None):
        """Integrate ``nsteps`` steps, recording scalar observables.

        Parameters
        ----------
        nsteps : int
            Number of integration steps.
        sample_every : int
            Record observables (and call ``callback``) every this many steps.
        callback : callable, optional
            Called as ``callback(system, time)`` at each sample, e.g. to
            accumulate a radial distribution function or histogram speeds.

        Returns
        -------
        dict of ndarray
            Time series of ``step``, ``time``, ``kinetic``, ``potential``,
            ``total_energy``, ``temperature`` and ``momentum`` (the last as an
            ``(nframes, d)`` array).
        """
        history = defaultdict(list)
        for _ in range(nsteps):
            self.step()
            if self.step_count % sample_every == 0:
                history["step"].append(self.step_count)
                history["time"].append(self.time)
                history["kinetic"].append(self.system.kinetic_energy)
                history["potential"].append(self.system.potential_energy)
                history["total_energy"].append(self.system.total_energy)
                history["temperature"].append(self.system.temperature)
                history["momentum"].append(self.system.momentum.copy())
                if callback is not None:
                    callback(self.system, self.time)
        out = {k: np.asarray(v) for k, v in history.items()}
        return out
