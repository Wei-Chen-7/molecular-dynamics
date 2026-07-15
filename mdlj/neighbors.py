"""Linked-cell list for O(N) neighbour finding under periodic boundaries.

Evaluating pair forces by looping over all ``N (N - 1) / 2`` pairs costs
``O(N**2)``.  Because the Lennard-Jones interaction is truncated at ``r_c``,
only nearby particles interact, so the work can be reduced to ``O(N)`` by
binning particles into a grid of cells of side ``>= r_c``.  A particle then
only interacts with partners in its own cell and the immediately neighbouring
cells.

This module produces *candidate* pairs (each unordered pair at most once);
the actual cutoff test and force evaluation happen in
:func:`mdlj.simulation.pair_forces`, so the cell list is guaranteed to yield
exactly the same forces and energies as the brute-force ``O(N**2)`` loop.

The implementation is fully vectorized: the only Python-level loop is over the
*constant* number of stencil offsets (5 in 2-D, 14 in 3-D), never over cells or
particle pairs.  Particles are bucketed by cell with a counting sort, and for
each offset the partners of every particle are gathered from the neighbouring
cell in one shot.

To keep the periodic bookkeeping unambiguous the cell list is only used when
there are at least three cells along every dimension (so that a cell and its
periodic image are never neighbours).  Otherwise the caller falls back to the
brute-force pair list, which is still correct.
"""

from __future__ import annotations

import itertools

import numpy as np

__all__ = ["all_pairs", "CellList"]


def all_pairs(n: int):
    """Return every unordered pair ``(i, j)`` with ``i < j`` as index arrays."""
    return np.triu_indices(n, k=1)


def _half_stencil(dim: int) -> np.ndarray:
    """Half of the ``3**dim`` neighbour offsets, so each cell pair is unique.

    Includes the zero offset (a cell paired with itself) plus every offset
    whose first non-zero component is positive.  This yields ``(3**dim - 1)/2``
    forward neighbours: 4 in 2-D and 13 in 3-D.
    """
    offsets = []
    for delta in itertools.product((-1, 0, 1), repeat=dim):
        if all(c == 0 for c in delta):
            offsets.append(delta)  # self cell
            continue
        for c in delta:  # keep only "forward" offsets
            if c != 0:
                if c > 0:
                    offsets.append(delta)
                break
    return np.array(offsets, dtype=int)


class CellList:
    """Linked-cell list over a periodic box.

    Parameters
    ----------
    box : array_like
        Box side length(s); a scalar (cube) or one value per dimension.
    rc : float
        Interaction cutoff; cells are made at least this wide.
    """

    def __init__(self, box, rc: float):
        self.box = np.atleast_1d(np.asarray(box, dtype=float))
        self.rc = float(rc)
        self.dim = self.box.size
        self.ncell = np.maximum(np.floor(self.box / self.rc).astype(int), 1)
        # A valid periodic cell list needs >= 3 cells per dimension.
        self.usable = bool(np.all(self.ncell >= 3))
        self._stencil = _half_stencil(self.dim)

    @staticmethod
    def _gather(source, target_flat, cell_start, cell_count, order):
        """Pairs (source[k], j) for every j in the cell ``target_flat[k]``.

        Fully vectorized expansion of variable-length neighbour blocks: each
        source particle is repeated by its partner count, and the partners are
        read out of the counting-sort order array in one gather.
        """
        counts = cell_count[target_flat]
        total = int(counts.sum())
        if total == 0:
            return (np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp))
        i = np.repeat(source, counts)
        # Offset of each output slot within its neighbour block.
        block_start = np.zeros(counts.size, dtype=np.intp)
        np.cumsum(counts[:-1], out=block_start[1:])
        within = np.arange(total) - np.repeat(block_start, counts)
        base = np.repeat(cell_start[target_flat], counts)
        j = order[base + within]
        return i, j

    def candidate_pairs(self, positions: np.ndarray):
        """Candidate interacting pairs ``(i, j)`` as two index arrays.

        Every unordered pair separated by less than ``rc`` (under the minimum
        image convention) appears exactly once.  Some returned pairs may be
        farther apart than ``rc``; they are filtered later by the force routine.
        """
        positions = np.asarray(positions, dtype=float)
        n = positions.shape[0]
        if not self.usable or n < 2:
            return all_pairs(n)

        cell_size = self.box / self.ncell
        coords = np.floor(positions / cell_size).astype(np.intp) % self.ncell
        flat = np.ravel_multi_index(coords.T, self.ncell)
        ncell_total = int(np.prod(self.ncell))

        # Counting sort: particles grouped by cell in `order`.
        cell_count = np.bincount(flat, minlength=ncell_total)
        cell_start = np.zeros(ncell_total, dtype=np.intp)
        np.cumsum(cell_count[:-1], out=cell_start[1:])
        order = np.argsort(flat, kind="stable").astype(np.intp)

        source = np.arange(n, dtype=np.intp)
        i_list = []
        j_list = []
        for delta in self._stencil:
            if not np.any(delta):
                # Same cell: keep each unordered pair once (i < j).
                i, j = self._gather(source, flat, cell_start, cell_count, order)
                keep = i < j
                i_list.append(i[keep])
                j_list.append(j[keep])
            else:
                neigh = (coords + delta) % self.ncell
                tflat = np.ravel_multi_index(neigh.T, self.ncell)
                i, j = self._gather(source, tflat, cell_start, cell_count, order)
                i_list.append(i)
                j_list.append(j)

        return np.concatenate(i_list), np.concatenate(j_list)
