"""Physics tests for the Lennard-Jones potential and pair force."""

import numpy as np
import pytest

from mdlj.potential import LennardJones, lj_force_magnitude, lj_potential


R_MIN = 2.0 ** (1.0 / 6.0)  # ~1.122462, location of the potential minimum


def test_potential_minimum_depth():
    """U(2**(1/6)) == -epsilon exactly (the well depth)."""
    assert lj_potential(R_MIN) == pytest.approx(-1.0, abs=1e-12)


def test_force_zero_at_minimum():
    """The force vanishes at the bottom of the well."""
    assert lj_force_magnitude(R_MIN) == pytest.approx(0.0, abs=1e-9)


def test_force_matches_finite_difference():
    """The analytic force equals -dU/dr from a central finite difference."""
    r = np.linspace(0.9, 2.4, 40)
    h = 1e-6
    fd_force = (lj_potential(r - h) - lj_potential(r + h)) / (2.0 * h)
    analytic = lj_force_magnitude(r)
    assert np.allclose(analytic, fd_force, rtol=1e-5)


def test_shift_makes_energy_continuous_at_cutoff():
    """The shifted potential is exactly zero at the cutoff and continuous."""
    lj = LennardJones(rc=2.5)
    # Exactly zero at and beyond the cutoff.
    assert lj.energy(lj.rc) == pytest.approx(0.0, abs=0.0)
    assert lj.energy(lj.rc + 0.1) == 0.0
    # Continuous: the limit from inside matches the value at the cutoff.
    just_inside = lj.energy(lj.rc - 1e-9)
    assert just_inside == pytest.approx(0.0, abs=1e-7)
    # The unshifted well at the cutoff is the documented ~-0.0163 epsilon.
    assert lj_potential(lj.rc) == pytest.approx(-0.0163169, abs=1e-6)


def test_shift_does_not_change_force():
    """A constant energy shift leaves the force inside the cutoff unchanged."""
    lj = LennardJones(rc=2.5)
    r = np.linspace(0.95, 2.4, 25)
    assert np.allclose(lj.force_magnitude(r), lj_force_magnitude(r))
