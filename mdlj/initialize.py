"""Initial conditions: lattice positions and Maxwell-Boltzmann velocities.

Particles are placed on a regular lattice (square or triangular in 2-D, FCC in
3-D) so that no two start closer than the steep repulsive core.  Velocities are
drawn from a Gaussian (Maxwell-Boltzmann) distribution at the target
temperature; the centre-of-mass velocity is then subtracted so the net momentum
is exactly zero, and the velocities are rescaled so the initial instantaneous
temperature exactly equals the requested value.

All quantities are in reduced Lennard-Jones units (``epsilon = sigma = m =
k_B = 1``).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "square_lattice",
    "triangular_lattice",
    "fcc_lattice",
    "initialize_positions",
    "maxwell_boltzmann_velocities",
]


def _box_from_density(n: int, density: float, dim: int) -> float:
    """Side length of a ``dim``-cube holding ``n`` particles at ``density``."""
    return (n / density) ** (1.0 / dim)


def square_lattice(n_side: int, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """``n_side**2`` particles on a square lattice; returns positions and box."""
    coords = (np.arange(n_side) + 0.5) * spacing
    gx, gy = np.meshgrid(coords, coords, indexing="ij")
    positions = np.column_stack([gx.ravel(), gy.ravel()])
    box = np.array([n_side * spacing, n_side * spacing])
    return positions, box


def triangular_lattice(n_side: int, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """A close-packed (triangular) 2-D lattice.

    Rows are offset by half a spacing and compressed vertically by
    ``sqrt(3)/2`` so the box stays periodic and the packing is isotropic.
    Returns ``2 * n_side**2`` particles and the box.
    """
    dx = spacing
    dy = spacing * np.sqrt(3.0) / 2.0
    positions = []
    for row in range(2 * n_side):
        offset = 0.5 * dx * (row % 2)
        for col in range(n_side):
            positions.append([col * dx + offset + 0.25 * dx, row * dy + 0.5 * dy])
    positions = np.array(positions)
    box = np.array([n_side * dx, 2 * n_side * dy])
    return positions, box


def fcc_lattice(n_cells: int, a: float) -> tuple[np.ndarray, np.ndarray]:
    """Face-centred-cubic lattice of ``4 * n_cells**3`` particles.

    Parameters
    ----------
    n_cells : int
        Number of unit cells along each axis.
    a : float
        Conventional cubic cell edge length.
    """
    basis = np.array([[0.0, 0.0, 0.0],
                      [0.5, 0.5, 0.0],
                      [0.5, 0.0, 0.5],
                      [0.0, 0.5, 0.5]])
    positions = []
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                cell = np.array([i, j, k])
                for b in basis:
                    positions.append((cell + b) * a)
    positions = np.array(positions)
    box = np.array([n_cells * a] * 3)
    return positions, box


def initialize_positions(n: int, density: float, dim: int = 2,
                         lattice: str = "square") -> tuple[np.ndarray, np.ndarray]:
    """Place ``n`` particles on a lattice at the requested number density.

    Parameters
    ----------
    n : int
        Number of particles.  For ``triangular`` it is rounded up to
        ``2 * n_side**2``; for ``fcc`` to ``4 * n_cells**3``; for ``square`` to
        ``n_side**dim``.
    density : float
        Target number density ``rho = N / V``.
    dim : int
        Spatial dimension (2 or 3).
    lattice : {"square", "triangular", "fcc"}
        Lattice type.  ``square`` works in 2-D or 3-D; ``triangular`` is 2-D
        only; ``fcc`` is 3-D only.

    Returns
    -------
    positions : ndarray, shape (N, dim)
    box : ndarray, shape (dim,)
        Box lengths, rescaled so that the final density is exactly ``density``.
    """
    lattice = lattice.lower()
    if lattice == "triangular":
        if dim != 2:
            raise ValueError("triangular lattice is 2-D only")
        n_side = int(np.ceil(np.sqrt(n / 2.0)))
        positions, box = triangular_lattice(n_side, 1.0)
    elif lattice == "fcc":
        if dim != 3:
            raise ValueError("fcc lattice is 3-D only")
        n_cells = int(np.ceil((n / 4.0) ** (1.0 / 3.0)))
        positions, box = fcc_lattice(n_cells, 1.0)
    elif lattice == "square":
        n_side = int(np.ceil(n ** (1.0 / dim)))
        if dim == 2:
            positions, box = square_lattice(n_side, 1.0)
        elif dim == 3:
            coords = (np.arange(n_side) + 0.5)
            gx, gy, gz = np.meshgrid(coords, coords, coords, indexing="ij")
            positions = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
            box = np.array([float(n_side)] * 3)
        else:
            raise ValueError("dim must be 2 or 3")
    else:
        raise ValueError(f"unknown lattice {lattice!r}")

    # Rescale the unit-spacing lattice so the density is exactly as requested.
    current_density = len(positions) / np.prod(box)
    scale = (current_density / density) ** (1.0 / dim)
    positions = positions * scale
    box = box * scale
    return positions, box


def maxwell_boltzmann_velocities(n: int, dim: int, temperature: float, rng,
                                 mass=1.0) -> np.ndarray:
    """Draw velocities from a Maxwell-Boltzmann distribution at ``temperature``.

    Each Cartesian component is Gaussian with variance ``k_B T / m``.  The
    centre-of-mass velocity is removed (zero net momentum) and the velocities
    are rescaled so the instantaneous temperature ``2 KE / (N_dof k_B)`` with
    ``N_dof = dim * n - dim`` is exactly ``temperature``.

    Parameters
    ----------
    n : int
        Number of particles.
    dim : int
        Spatial dimension.
    temperature : float
        Target temperature (in units of ``epsilon / k_B``).
    rng : numpy.random.Generator
        Seeded random generator for reproducibility.
    mass : float or ndarray
        Particle mass(es).
    """
    mass_arr = np.broadcast_to(np.asarray(mass, dtype=float), (n,)).copy()
    std = np.sqrt(temperature / mass_arr)[:, None]
    velocities = rng.normal(0.0, 1.0, size=(n, dim)) * std

    # Zero net momentum.
    v_cm = np.sum(mass_arr[:, None] * velocities, axis=0) / mass_arr.sum()
    velocities -= v_cm

    # Rescale to the exact target temperature.
    n_dof = dim * n - dim
    kinetic = 0.5 * np.sum(mass_arr[:, None] * velocities ** 2)
    current_t = 2.0 * kinetic / n_dof
    if current_t > 0:
        velocities *= np.sqrt(temperature / current_t)
    return velocities
