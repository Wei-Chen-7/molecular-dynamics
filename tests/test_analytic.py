"""Internal consistency of the closed-form analytic references."""

import numpy as np
from scipy import integrate

from mdlj import analytic


def test_speed_pdf_normalized():
    """The Maxwell-Boltzmann speed PDF integrates to 1 in 2-D and 3-D."""
    for dim in (2, 3):
        area, _ = integrate.quad(
            lambda v: analytic.maxwell_boltzmann_speed_pdf(v, 1.3, dim), 0, 20
        )
        assert abs(area - 1.0) < 1e-6


def test_speed_moments_match_pdf_integrals():
    """The tabulated mean and rms speeds match integrals of the PDF."""
    T, dim = 1.3, 3
    pdf = lambda v: analytic.maxwell_boltzmann_speed_pdf(v, T, dim)
    mean_int, _ = integrate.quad(lambda v: v * pdf(v), 0, 25)
    msq_int, _ = integrate.quad(lambda v: v * v * pdf(v), 0, 25)
    assert abs(mean_int - analytic.mean_speed(T, dim)) < 1e-6
    assert abs(np.sqrt(msq_int) - analytic.rms_speed(T, dim)) < 1e-6


def test_speed_cdf_matches_pdf():
    """The analytic CDF is the running integral of the PDF."""
    T, dim = 0.8, 2
    for v in (0.5, 1.0, 1.7, 2.5):
        integral, _ = integrate.quad(
            lambda x: analytic.maxwell_boltzmann_speed_pdf(x, T, dim), 0, v
        )
        assert abs(integral - analytic.maxwell_boltzmann_speed_cdf(v, T, dim)) < 1e-6


def test_speed_ordering_and_ratios():
    """v_p < <v> < v_rms with the textbook dimensional ratios."""
    for dim in (2, 3):
        T = 1.1
        vp = analytic.most_probable_speed(T, dim)
        vmean = analytic.mean_speed(T, dim)
        vrms = analytic.rms_speed(T, dim)
        assert vp < vmean < vrms
        assert analytic.rms_speed(T, dim) == np.sqrt(dim * T)
