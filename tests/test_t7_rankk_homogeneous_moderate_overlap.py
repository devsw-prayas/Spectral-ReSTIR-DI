"""Point-probe for T7 (session_log_restir_3 V-B, moderate overlap: rank-2
homogeneous trichotomy, "inner-filter bias mechanism").

Historical V-B: candidates (species j ~ categorical(w_j^A), lambda' ~
N(mu_j^A, sigma_j^A)) generated under vertex-context A, reused at vertex
B. Moderate-overlap mixtures A=[0.97,0.03] -> B=[0.09,0.91]. Naive +12.71%
biased (decisive, z=+380), IS-reweight unbiased but ~12x variance-degraded,
detached fix clean. Finding #2 ("inner-filter effect"): the bias is
carried entirely by DIFFERENTIAL TRANSMITTANCE between the species'
absorption bands -- exactly zero in the transparent-medium limit, and
grows with optical depth (+12.7% at tau~O(1) -> +29.5% at 3x thickness).

This probe reconstructs that mechanism exactly (fresh concrete parameters,
per the session log's own "band parameters are reconstructed comparables,
not mirrors" note -- session_log_restir_3.md sec 0), using
`heterogeneous_lookup.py` (module 6) at exact-quadrature precision (no MC,
same discipline as that module's own test file): two always-positive,
oppositely-graded species concentration profiles (homogeneous in the sense
that neither species is ever exactly absent -- no hard support boundary,
that is T9/T10's territory), a light-importance-only candidate-generation
weight that (per the module docstring's "naive's cancellation trap")
deliberately ignores the medium's self-absorption transmittance term that
the true local target carries. Confirms all three legs of finding #2: near
zero at L->0 (transparent), decisive at L~O(1) (moderate), and roughly 3x
larger at 3x the path length.
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from heterogeneous_lookup import (
    species_weight,
    local_target,
    naive_score,
    is_reweight_score,
    fix_local_score,
)

torch.set_default_dtype(torch.float64)

LAMBDA = torch.linspace(400.0, 700.0, 4000)
DLAM = (LAMBDA[1] - LAMBDA[0]).item()
LAMBDA_S = 590.0
MU = {0: 560.0, 1: 620.0}
SIGMA = {0: 40.0, 1: 40.0}
EMISSION_MU, EMISSION_SIGMA = 590.0, 70.0
STEEPNESS = 15.0  # controls how sharply the mixture flips between z_A and z_B
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
    # both species always positive everywhere -- homogeneous trichotomy
    # family, no support boundary (contrast T9/T10's hard cutoff field)
    return (1.0 + STEEPNESS * z) if j == 0 else (1.0 + STEEPNESS * (1.0 - z))


def _column_density(j, z):
    # closed-form integral_z^1 conc(j, z') dz' (remaining path to the far boundary)
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
    return K_NO_TRANS[j]  # candidate generation ignores the transmittance term


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


def _mix(z):
    weights = {k: species_weight(conc, excitation, _no_trans_integral, k, z, LAMBDA_S) for k in (0, 1)}
    total = weights[0] + weights[1]
    return weights[0] / total, weights[1] / total


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


def test_mixture_flips_moderately_between_contexts():
    mix_a = _mix(Z_A)
    mix_b = _mix(Z_B)
    assert mix_a[0] < 0.2 and mix_a[1] > 0.8
    assert mix_b[0] > 0.8 and mix_b[1] < 0.2


def _naive_rel_err(L_scale):
    target_fn = _make_target(L_scale)
    truth = _quadrature(target_fn, Z_B)
    naive = _expected_score(lambda y: naive_score(target_fn, proposal_pdf_fn, y, Z_B), Z_A)
    return (naive - truth) / truth


def test_naive_biased_at_moderate_optical_depth():
    rel_err = _naive_rel_err(L_scale=30.0)
    assert rel_err > 0.05  # decisively nonzero, moderate-overlap magnitude


def test_naive_bias_vanishes_in_transparent_limit():
    rel_err = _naive_rel_err(L_scale=0.0)
    assert abs(rel_err) < 1e-3  # transmittance -> 1 uniformly, cancellation trap applies exactly


def test_naive_bias_grows_with_optical_depth():
    moderate = _naive_rel_err(L_scale=30.0)
    thicker = _naive_rel_err(L_scale=90.0)  # 3x path length
    assert thicker > 2.0 * moderate  # grows well beyond linear-in-L, matches historical +12.7%->+29.5%


def test_is_reweight_and_fix_local_unbiased_at_moderate_optical_depth():
    target_fn = _make_target(L_scale=30.0)
    truth = _quadrature(target_fn, Z_B)

    isr = _expected_score(lambda y: is_reweight_score(target_fn, proposal_pdf_fn, y, Z_A, Z_B), Z_A)
    assert abs((isr - truth) / truth) < 1e-9

    fixl = _expected_score(lambda y: fix_local_score(target_fn, proposal_pdf_fn, y, Z_B), Z_B)
    assert abs((fixl - truth) / truth) < 1e-9


if __name__ == "__main__":
    test_mixture_flips_moderately_between_contexts()
    test_naive_biased_at_moderate_optical_depth()
    test_naive_bias_vanishes_in_transparent_limit()
    test_naive_bias_grows_with_optical_depth()
    test_is_reweight_and_fix_local_unbiased_at_moderate_optical_depth()
    print("all T7 tests passed")
