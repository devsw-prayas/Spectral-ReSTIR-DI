"""T-tier point-probes for module 6 (heterogeneous_lookup.py).

Rank-2 species mixture, covering the A6 lookup trichotomy. Every
expectation below is computed by
exact quadrature over a fixed lambda' grid (no Monte Carlo, no RNG) -- same
discipline as `free_path_pdf`'s quadrature ground truth, chosen because the
trichotomy's claims are about *expectations*, which quadrature gives exactly
rather than approximately.

Two separate toy fields, matched to what each case needs:
- A smooth, always-positive concentration field (`CONC_SMOOTH`) with a
  transmittance term the candidate-generation weight deliberately does NOT
  track (see heterogeneous_lookup.py's "naive's cancellation trap" docstring
  note) -- this is what makes naive genuinely biased instead of accidentally
  exact. Used for the naive-biased / IS-reweight-safe-and-unbiased cases.
- A hard-cutoff field (`CONC_CUTOFF`, species 0 dies at z>=0.5) for the
  support-violation case (T10's -78.4% mechanism).

Covers all three A6 cases: naive biased under safe support (case 1),
IS-reweight exactly unbiased under safe support (case 2, containment
holds), IS-reweight structurally biased under a support violation (case 2,
containment fails), and fix-local exactly unbiased regardless (case 3).
"""

import sys
import os
import math
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from heterogeneous_lookup import (
    species_weight,
    local_target,
    naive_score,
    is_reweight_score,
    fix_local_score,
    has_support_violation,
    lookup_trichotomy_case,
)

torch.set_default_dtype(torch.float64)

LAMBDA = torch.linspace(400.0, 700.0, 4000)
DLAM = (LAMBDA[1] - LAMBDA[0]).item()
LAMBDA_S = 590.0
MU = {0: 500.0, 1: 600.0}
SIGMA = {0: 15.0, 1: 40.0}
E = {0: 1.0, 1: 1.3}
EMISSION_MU, EMISSION_SIGMA = 550.0, 60.0


def _gaussian_pdf(x, mu, sigma):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


A_TENSOR = {j: _gaussian_pdf(LAMBDA, MU[j], SIGMA[j]) for j in (0, 1)}
LE_TENSOR = _gaussian_pdf(LAMBDA, EMISSION_MU, EMISSION_SIGMA)
# candidate generation's own per-species normalizer -- deliberately the
# "vacuum" integral(a_j*L_e), NOT transmittance-aware (see module docstring)
K_NO_TRANS = {j: torch.trapz(A_TENSOR[j] * LE_TENSOR, LAMBDA).item() for j in (0, 1)}


def absorption(j, lam):
    return _gaussian_pdf(torch.as_tensor(float(lam)), MU[j], SIGMA[j]).item()


def emission(lam):
    return _gaussian_pdf(torch.as_tensor(float(lam)), EMISSION_MU, EMISSION_SIGMA).item()


def excitation(j, lambda_s):
    return E[j]


# --- smooth, always-positive field: species 0 rising, species 1 falling ---
def conc_smooth(j, z):
    return 1.0 + 9.0 * z if j == 0 else 10.0 - 9.0 * z


def _column_density_smooth(j, z):
    # closed-form integral_z^1 conc_smooth(j, z') dz'
    if j == 0:
        f = lambda zz: zz + 4.5 * zz * zz
    else:
        f = lambda zz: 10.0 * zz - 4.5 * zz * zz
    return f(1.0) - f(z)


def transmittance_smooth(j, lam_prime, z):
    optical_depth = sum(
        absorption(k, lam_prime) * _column_density_smooth(k, z) for k in (0, 1)
    )
    return math.exp(-optical_depth)


def target_smooth(y, z):
    j, lam = y
    return local_target(
        conc_smooth, excitation, absorption, emission, j, lam, z, LAMBDA_S,
        transmittance_fn=transmittance_smooth,
    )


def _no_trans_integral(j, z):
    return K_NO_TRANS[j]  # candidate generation ignores the transmittance term


def proposal_pdf_smooth(y, z):
    j, lam = y
    weights = {
        k: species_weight(conc_smooth, excitation, _no_trans_integral, k, z, LAMBDA_S)
        for k in (0, 1)
    }
    total = weights[0] + weights[1]
    if total == 0.0:
        return 0.0
    return (weights[j] / total) * absorption(j, lam)


# --- hard-cutoff field: species 0 dies at z >= 0.5 ---
def conc_cutoff(j, z):
    if j == 0:
        return max(0.0, 1.0 - 2.0 * z)
    return 1.0


def _integral_a_le(j):
    return torch.trapz(A_TENSOR[j] * LE_TENSOR, LAMBDA).item()


INTEGRAL_A_LE_CUTOFF = {j: _integral_a_le(j) for j in (0, 1)}


def target_cutoff(y, z):
    j, lam = y
    return local_target(conc_cutoff, excitation, absorption, emission, j, lam, z, LAMBDA_S)


def proposal_pdf_cutoff(y, z):
    j, lam = y
    weights = {
        k: species_weight(conc_cutoff, excitation, lambda k, z: INTEGRAL_A_LE_CUTOFF[k], k, z, LAMBDA_S)
        for k in (0, 1)
    }
    total = weights[0] + weights[1]
    if total == 0.0:
        return 0.0
    return (weights[j] / total) * absorption(j, lam)


def _quadrature(fn, z):
    # sum_j integral_lambda fn((j,lambda), z) dlambda, over the fixed LAMBDA grid
    total = 0.0
    for j in (0, 1):
        total += sum(fn((j, lam.item()), z) for lam in LAMBDA) * DLAM
    return total


def _expected_score(score_fn, proposal_pdf_fn, z_a):
    # E_{y~q_zA}[score(y)] = sum_j integral score((j,lambda)) * q_zA((j,lambda)) dlambda
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


def test_naive_biased_under_safe_support():
    z_a, z_b = 0.1, 0.9  # smooth field: both species alive everywhere
    expected_naive = _expected_score(
        lambda y: naive_score(target_smooth, proposal_pdf_smooth, y, z_b), proposal_pdf_smooth, z_a
    )
    truth = _quadrature(target_smooth, z_b)
    rel_err = abs(expected_naive - truth) / truth
    assert rel_err > 1e-3  # decisively nonzero vs. quadrature's ~1e-9 precision floor


def test_is_reweight_unbiased_under_safe_support():
    z_a, z_b = 0.1, 0.9
    expected_is = _expected_score(
        lambda y: is_reweight_score(target_smooth, proposal_pdf_smooth, y, z_a, z_b), proposal_pdf_smooth, z_a
    )
    truth = _quadrature(target_smooth, z_b)
    assert abs(expected_is - truth) / truth < 1e-9


def test_is_reweight_structurally_biased_under_support_violation():
    z_a, z_b = 0.6, 0.1  # cutoff field: species 0 dead at z_a, alive at z_b
    assert has_support_violation(conc_cutoff, 0, z_a, z_b)
    assert not has_support_violation(conc_cutoff, 1, z_a, z_b)
    assert lookup_trichotomy_case(conc_cutoff, (0, MU[0]), z_a, z_b) == "support_violation"
    assert lookup_trichotomy_case(conc_cutoff, (1, MU[1]), z_a, z_b) == "reweight_safe"

    expected_is = _expected_score(
        lambda y: is_reweight_score(target_cutoff, proposal_pdf_cutoff, y, z_a, z_b), proposal_pdf_cutoff, z_a
    )
    truth = _quadrature(target_cutoff, z_b)
    missing_species0 = sum(target_cutoff((0, lam.item()), z_b) for lam in LAMBDA) * DLAM

    # species 0's entire contribution is silently dropped (existence
    # failure, not a phantom zero) -- estimator equals truth minus exactly
    # species 0's share, not just "some" bias
    assert abs(expected_is - (truth - missing_species0)) < 1e-9
    assert (truth - expected_is) / truth > 0.1  # decisive, matches T10's magnitude


def test_fix_local_unbiased_regardless_of_source():
    z_b = 0.9
    expected_fixlocal = _expected_score(
        lambda y: fix_local_score(target_smooth, proposal_pdf_smooth, y, z_b), proposal_pdf_smooth, z_b
    )
    truth = _quadrature(target_smooth, z_b)
    assert abs(expected_fixlocal - truth) / truth < 1e-9


if __name__ == "__main__":
    test_naive_biased_under_safe_support()
    test_is_reweight_unbiased_under_safe_support()
    test_is_reweight_structurally_biased_under_support_violation()
    test_fix_local_unbiased_regardless_of_source()
    print("all heterogeneous_lookup tests passed")
