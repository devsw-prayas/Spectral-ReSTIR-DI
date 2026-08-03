"""Point-probe for T11 (session_log_restir_3 V-D(b): spectral-separation
control, "coupling collapse").

Historical V-D found a *bounded, smooth* coupling channel between species
at a fixed vertex: perturbing a species-1 band parameter (e.g. its
absorption-band center) contaminates species 2's own emission integral
purely through the SHARED transmittance term (inner-filter effect, same
mechanism as T7/T9's naive bias, here viewed as a sensitivity rather than a
reuse-estimator question). The control that pins this down as genuinely
spectral (not some other artifact): move species 1's band well clear of
species 2's band and the light spectrum's support, and the coupling
derivative collapses by orders of magnitude (historical: 150x for the band
center, 800x for a second parameter) -- "same gating knob as the surface
case" (spectral-separation control also appears in the rank-k surface
literature this project builds on).

This probe reconstructs the mechanism with a fixed vertex (no z-dependence
needed -- V-D's edge-position parameter is dropped here; the band-center
control alone is the direct, cheap version of the claim) and a two-sided
finite-difference derivative of species 2's own local-target integral
w.r.t. species 1's absorption-band center `mu1`, checked at two step sizes
for FD stability (V9 discipline, matching the project's closed-form-inner-
integral convention). Fresh concrete parameters, not the historical exact
numbers (`session_log_restir_3.md` sec 0's standing "not recoverable"
caveat) -- the collapse ratio here comes out far larger than the historical
150x/800x, which is expected: a fresh reconstruction with different band
widths/light-spectrum support has no reason to reproduce the same
magnitude, only the same qualitative "collapses by orders of magnitude"
claim.
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from heterogeneous_lookup import local_target

torch.set_default_dtype(torch.float64)

LAMBDA = torch.linspace(300.0, 800.0, 3000)
DLAM = (LAMBDA[1] - LAMBDA[0]).item()
LAMBDA_S = 590.0
MU2, SIGMA2 = 620.0, 40.0  # species 2 -- the one whose sensitivity we measure
SIGMA1 = 40.0  # species 1's band width (its center `mu1` is the swept parameter)
EMISSION_MU, EMISSION_SIGMA = 590.0, 70.0
C1, C2 = 1.0, 1.0
Z_FIXED = 0.5  # fixed vertex -- this control isolates the spectral channel, not a spatial one
L_SCALE = 20.0  # shared self-absorption strength


def _gaussian_pdf(x, mu, sigma):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def absorption(j, lam, mu1):
    mu = mu1 if j == 0 else MU2
    sigma = SIGMA1 if j == 0 else SIGMA2
    return _gaussian_pdf(torch.as_tensor(float(lam)), mu, sigma).item()


def emission(lam):
    return _gaussian_pdf(torch.as_tensor(float(lam)), EMISSION_MU, EMISSION_SIGMA).item()


def conc(j, z):
    return C1 if j == 0 else C2


def excitation(j, lambda_s):
    return 1.0


def _transmittance(j, lam_prime, z, mu1):
    # shared optical depth through BOTH species -- the only channel by
    # which species 1's parameters can influence species 2's own integral
    optical_depth = L_SCALE * (
        conc(0, z) * absorption(0, lam_prime, mu1) + conc(1, z) * absorption(1, lam_prime, mu1)
    )
    return math.exp(-optical_depth)


def I2(mu1):
    """Species 2's own local-target integral over lambda', at fixed z, as a
    function of species 1's band center `mu1` (entering only through the
    shared transmittance term)."""
    total = 0.0
    for lam in LAMBDA:
        lam = lam.item()
        total += local_target(
            conc, excitation, lambda j, l: absorption(j, l, mu1), emission,
            1, lam, Z_FIXED, LAMBDA_S,
            transmittance_fn=lambda j, l, z: _transmittance(j, l, z, mu1),
        ) * DLAM
    return total


def _fd_derivative(mu1_center, h):
    return (I2(mu1_center + h) - I2(mu1_center - h)) / (2 * h)


MU1_CLOSE = 560.0  # species 1 overlapping species 2 / the light spectrum's support
MU1_FAR = 350.0    # species 1 moved well clear of both


def test_fd_derivative_is_stable_across_step_sizes():
    d1 = _fd_derivative(MU1_CLOSE, 5.0)
    d2 = _fd_derivative(MU1_CLOSE, 2.5)
    assert abs(d1 - d2) / abs(d1) < 0.01  # converged, not an FD-step artifact


def test_coupling_is_bounded_but_nonzero_when_spectrally_close():
    d = _fd_derivative(MU1_CLOSE, 5.0)
    assert math.isfinite(d)
    assert abs(d) > 1e-8  # a real, measurable cross-species sensitivity


def test_coupling_collapses_under_spectral_separation_control():
    d_close = _fd_derivative(MU1_CLOSE, 5.0)
    d_far = _fd_derivative(MU1_FAR, 5.0)
    assert abs(d_far) < abs(d_close)
    ratio = abs(d_close) / abs(d_far)
    assert ratio > 100.0  # orders-of-magnitude collapse, matches the historical 150x/800x class


if __name__ == "__main__":
    test_fd_derivative_is_stable_across_step_sizes()
    test_coupling_is_bounded_but_nonzero_when_spectrally_close()
    test_coupling_collapses_under_spectral_separation_control()
    print("all T11 tests passed")
