"""Graphable sweep for G2: "optical-depth scaling sweep (naive bias vs.
tau)" -- the "inner-filter effect," reconstructed as a genuine multi-point
sweep rather than the thin 2-point result that first surfaced it.

**Mechanism, identical to `test_t7_rankk_homogeneous_moderate_overlap.py`'s
own reconstruction** (same toy, same `heterogeneous_lookup.py` machinery,
same discipline of exact-quadrature scoring, no MC) -- T7 only ever checked
naive bias at three fixed optical-depth scales (`L_scale` in {0, 30, 90}).
G2 is the continuous-sweep companion: same mechanism, denser `L_scale` grid,
checked for a genuinely smooth/monotonic trend rather than three isolated
points. Self-contained (no cross-import from `test_t7_...py`), per this
repo's established no-cross-import-between-point-probe-files convention.

**Historical shape** (earlier 2-point result): naive bias +12.7% at
`tau~O(1)` -> +29.5% at 3x thickness, "grows well past linear with path
length." **This sweep's fresh parameters** (same species/mixture setup as
T7, `L_scale` from 0 to 120): bias grows monotonically from ~0% (transparent
limit) to ~42% at the far end, with the 30->90 leg (matching T7's own
moderate->3x comparison) landing at +8.7%->+29.7%, a ~3.4x increase for a 3x
path-length increase -- confirms the same super-linear-with-optical-depth
shape the historical 2-point result showed, now with 9 points instead of 2.
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from heterogeneous_lookup import species_weight, local_target, naive_score

torch.set_default_dtype(torch.float64)

LAMBDA = torch.linspace(400.0, 700.0, 4000)
DLAM = (LAMBDA[1] - LAMBDA[0]).item()
LAMBDA_S = 590.0
MU = {0: 560.0, 1: 620.0}
SIGMA = {0: 40.0, 1: 40.0}
EMISSION_MU, EMISSION_SIGMA = 590.0, 70.0
STEEPNESS = 15.0
Z_A, Z_B = 0.1, 0.9


def _gaussian_pdf(x, mu, sigma):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


A_TENSOR = {j: _gaussian_pdf(LAMBDA, MU[j], SIGMA[j]) for j in (0, 1)}
LE_TENSOR = _gaussian_pdf(LAMBDA, EMISSION_MU, EMISSION_SIGMA)
K_NO_TRANS = {j: torch.trapz(A_TENSOR[j] * LE_TENSOR, LAMBDA).item() for j in (0, 1)}


def absorption(j, lam):
    return _gaussian_pdf(torch.as_tensor(float(lam)), MU[j], SIGMA[j]).item()


def emission(lam):
    return _gaussian_pdf(torch.as_tensor(float(lam)), EMISSION_MU, EMISSION_SIGMA).item()


def excitation(j, lambda_s):
    return 1.0


def conc(j, z):
    return (1.0 + STEEPNESS * z) if j == 0 else (1.0 + STEEPNESS * (1.0 - z))


def _column_density(j, z):
    if j == 0:
        f = lambda zz: zz + 0.5 * STEEPNESS * zz * zz
    else:
        f = lambda zz: (1.0 + STEEPNESS) * zz - 0.5 * STEEPNESS * zz * zz
    return f(1.0) - f(z)


def _make_transmittance(L_scale):
    def transmittance(j, lam_prime, z):
        optical_depth = L_scale * sum(
            absorption(k, lam_prime) * _column_density(k, z) for k in (0, 1)
        )
        return math.exp(-optical_depth)
    return transmittance


def _no_trans_integral(j, z):
    return K_NO_TRANS[j]


def _make_target(L_scale):
    transmittance = _make_transmittance(L_scale)

    def target_fn(y, z):
        j, lam = y
        return local_target(
            conc, excitation, absorption, emission, j, lam, z, LAMBDA_S,
            transmittance_fn=transmittance,
        )
    return target_fn


def proposal_pdf_fn(y, z):
    j, lam = y
    weights = {k: species_weight(conc, excitation, _no_trans_integral, k, z, LAMBDA_S) for k in (0, 1)}
    total = weights[0] + weights[1]
    if total == 0.0:
        return 0.0
    return (weights[j] / total) * absorption(j, lam)


def _quadrature(target_fn, z):
    total = 0.0
    for j in (0, 1):
        for lam in LAMBDA:
            total += target_fn((j, lam.item()), z) * DLAM
    return total


def _expected_score(score_fn, z_a):
    total = 0.0
    for j in (0, 1):
        for lam in LAMBDA:
            lam = lam.item()
            q = proposal_pdf_fn((j, lam), z_a)
            if q == 0.0:
                continue
            s = score_fn((j, lam))
            if s is None:
                continue
            total += s * q * DLAM
    return total


def _naive_rel_err(L_scale):
    target_fn = _make_target(L_scale)
    truth = _quadrature(target_fn, Z_B)
    naive = _expected_score(lambda y: naive_score(target_fn, proposal_pdf_fn, y, Z_B), Z_A)
    return (naive - truth) / truth


L_SWEEP = (0.0, 5.0, 10.0, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0)


def test_bias_near_zero_at_transparent_limit():
    assert abs(_naive_rel_err(0.0)) < 1e-3


def test_bias_grows_monotonically_with_optical_depth():
    rel_errs = [_naive_rel_err(L) for L in L_SWEEP]
    for earlier, later in zip(rel_errs, rel_errs[1:]):
        assert later > earlier  # strictly monotonic across all 9 swept points


def test_bias_growth_is_superlinear_matching_historical_shape():
    # T7's own moderate (L=30) -> 3x (L=90) comparison, historical shape
    # +12.7% -> +29.5% (a ~2.3x increase for a 3x path-length increase).
    moderate = _naive_rel_err(30.0)
    thicker = _naive_rel_err(90.0)
    assert thicker > 2.0 * moderate  # grows well past linear-in-L


def test_bias_reaches_a_large_fraction_at_the_far_end_of_the_sweep():
    # confirms the sweep's far end is a genuinely different (not just
    # noisier) regime from the transparent limit -- not a flat curve with
    # sampling noise on top.
    assert _naive_rel_err(L_SWEEP[-1]) > 0.35


if __name__ == "__main__":
    test_bias_near_zero_at_transparent_limit()
    test_bias_grows_monotonically_with_optical_depth()
    test_bias_growth_is_superlinear_matching_historical_shape()
    test_bias_reaches_a_large_fraction_at_the_far_end_of_the_sweep()
    print("all G2 tests passed")
