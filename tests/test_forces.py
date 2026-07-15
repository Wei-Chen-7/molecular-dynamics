"""Physics tests for pair forces, PBC/minimum image and the cell list."""

import numpy as np

from mdlj import LennardJones, System, initialize_positions
from mdlj.neighbors import CellList, all_pairs
from mdlj.observables import minimum_image_distances
from mdlj.simulation import pair_forces


def _random_system(n, rho, dim, lattice, seed):
    rng = np.random.default_rng(seed)
    pos, box = initialize_positions(n, density=rho, dim=dim, lattice=lattice)
    pos = (pos + rng.normal(0.0, 0.05, pos.shape)) % box
    return pos, box


def test_total_force_sums_to_zero():
    """Internal pair forces cancel: the net force on the system is zero."""
    pos, box = _random_system(150, 0.6, 2, "square", seed=0)
    forces, _, _ = pair_forces(pos, box, LennardJones(), *all_pairs(len(pos)))
    assert np.allclose(forces.sum(axis=0), 0.0, atol=1e-10)


def test_newtons_third_law():
    """The force from pair (i, j) is equal and opposite for i and j."""
    pos, box = _random_system(40, 0.5, 2, "square", seed=1)
    lj = LennardJones()
    i, j = np.array([3, 7, 11]), np.array([15, 22, 30])
    # Force each single pair contributes, computed in isolation.
    for a, b in zip(i, j):
        f_full, _, _ = pair_forces(pos, box, lj, np.array([a]), np.array([b]))
        # Only particles a and b feel a force; they are opposite.
        assert np.allclose(f_full[a], -f_full[b], atol=1e-12)


def test_minimum_image_within_half_box():
    """Every minimum-image displacement component lies in [-L/2, L/2]."""
    rng = np.random.default_rng(2)
    box = np.array([8.0, 8.0])
    pos = rng.uniform(-20, 20, size=(60, 2))  # deliberately outside the box
    i, j = all_pairs(len(pos))
    rij = pos[i] - pos[j]
    rij -= box * np.round(rij / box)
    assert np.all(rij >= -box / 2 - 1e-12)
    assert np.all(rij <= box / 2 + 1e-12)


def test_cell_list_matches_bruteforce_2d():
    """Cell-list forces, energy and virial match the O(N^2) result (2-D)."""
    pos, box = _random_system(169, 0.7, 2, "square", seed=3)
    lj = LennardJones()
    cl = CellList(box, lj.rc)
    assert cl.usable
    fb, peb, vb = pair_forces(pos, box, lj, *all_pairs(len(pos)))
    fc, pec, vc = pair_forces(pos, box, lj, *cl.candidate_pairs(pos))
    assert np.allclose(fb, fc, atol=1e-10)
    assert abs(peb - pec) < 1e-10
    assert abs(vb - vc) < 1e-10


def test_cell_list_matches_bruteforce_3d():
    """Cell-list forces, energy and virial match the O(N^2) result (3-D)."""
    pos, box = _random_system(256, 0.5, 3, "fcc", seed=4)
    lj = LennardJones()
    cl = CellList(box, lj.rc)
    assert cl.usable  # box large enough for >= 3 cells per dimension
    fb, peb, vb = pair_forces(pos, box, lj, *all_pairs(len(pos)))
    fc, pec, vc = pair_forces(pos, box, lj, *cl.candidate_pairs(pos))
    assert np.allclose(fb, fc, atol=1e-10)
    assert abs(peb - pec) < 1e-10
    assert abs(vb - vc) < 1e-10


def test_system_forces_agree_between_paths():
    """A System using the cell list agrees with one using brute force."""
    pos, box = _random_system(200, 0.7, 2, "square", seed=5)
    vel = np.zeros_like(pos)
    s_cells = System(pos, vel, box, use_cell_list=True)
    s_brute = System(pos, vel, box, use_cell_list=False)
    assert s_cells.use_cell_list  # actually exercised
    assert np.allclose(s_cells.forces, s_brute.forces, atol=1e-10)
    assert abs(s_cells.potential_energy - s_brute.potential_energy) < 1e-10
