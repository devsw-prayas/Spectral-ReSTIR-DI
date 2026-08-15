"""Point-probe for T8 (well-separated: rank-2 homogeneous trichotomy,
"silent support collapse").

Historical V-B, well-separated mixtures ([1,0] -> [0,1]): naive lands
near-zero error "by cancellation" (a trap, not correctness), while
IS-reweight suffers a *silent* failure -- w_2^A underflows to ~0, species 2
is never sampled, its entire contribution goes missing, and the reported
error bar is tiny (confidently wrong, not just high-variance).

This probe reconstructs both halves at exact-quadrature precision for the
"near-cancellation" and "formally still unbiased" claims, and via real MC
sampling for the "silent" half specifically -- quadrature alone cannot show
a finite-sample phenomenon; you need actual draws to see IS-reweight's
sample composition go to *exactly* zero species-0 draws and watch its
reported standard error collapse right along with the (wrong) mean. Same
species/light bands and inner-filter-free target as T7
(`tests/test_t7_rankk_homogeneous_moderate_overlap.py`), just with an
extreme concentration gradient (species 0 confined to z~0, species 1 to
z~1) so context A (z_A=0) and context B (z_B=1) are essentially pure single
species with opposite identity -- reconstructed, not the historical exact
numbers (the original parameters were already flagged as non-recoverable).
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
)

torch.set_default_dtype(torch.float64)

LAMBDA = torch.linspace(400.0, 700.0, 4000)
DLAM = (LAMBDA[1] - LAMBDA[0]).item()
LAMBDA_S = 590.0
MU = {0: 560.0, 1: 620.0}
SIGMA = {0: 40.0, 1: 40.0}
EMISSION_MU, EMISSION_SIGMA = 590.0, 70.0
STEEPNESS = 1.0e7  # pushes mixA/mixB to ~single-species purity
Z_A, Z_B = 0.0, 1.0


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
    # same family as T7, but STEEPNESS pushed extreme -- species 0 is
    # effectively confined to z~0, species 1 to z~1, while remaining
    # formally positive everywhere (no hard support boundary; T10 covers
    # the genuine hard-cutoff case)
    return (1.0 + STEEPNESS * z) if j == 0 else (1.0 + STEEPNESS * (1.0 - z))


def _no_trans_integral(j, z):
    return K_NO_TRANS[j]


def target_fn(y, z):
    j, lam = y
    return local_target(conc, excitation, absorption, emission, j, lam, z, LAMBDA_S)


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


def _quadrature(z):
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


TRUTH_B = _quadrature(Z_B)


def test_mixtures_are_essentially_pure_and_opposite():
    mix_a = _mix(Z_A)
    mix_b = _mix(Z_B)
    assert mix_a[1] > 1.0 - 1e-4 and mix_a[0] < 1e-4  # z_A~0: species 1 dominant, species 0 rare
    assert mix_b[0] > 1.0 - 1e-4 and mix_b[1] < 1e-4  # z_B~1: species 0 dominant, species 1 rare


def test_naive_near_unbiased_by_cancellation_not_correctness():
    """At full separation each context is ~single-species, so naive's
    formula degenerates to (approximately) the correctly-normalized
    single-species estimator -- near-zero error, but structurally still the
    wrong formula (T7 already showed it's decisively biased at moderate
    separation with the identical mechanism)."""
    naive = _expected_score(lambda y: naive_score(target_fn, proposal_pdf_fn, y, Z_B), Z_A)
    assert abs((naive - TRUTH_B) / TRUTH_B) < 1e-2


def test_is_reweight_exactly_unbiased_in_the_infinite_sample_limit():
    """The field is smooth (never exactly zero), so IS-reweight's formal
    expectation (exact quadrature) stays unbiased even though its practical
    support has collapsed to a razor-thin sliver -- the "silent" part is a
    finite-sample phenomenon, not a broken expectation (see the MC test
    below)."""
    isr = _expected_score(lambda y: is_reweight_score(target_fn, proposal_pdf_fn, y, Z_A, Z_B), Z_A)
    assert abs((isr - TRUTH_B) / TRUTH_B) < 1e-6


def _sample_species_and_lambda(z, rng):
    w0, _ = _mix(z)
    j = 0 if torch.rand((), generator=rng).item() < w0 else 1
    lam = torch.normal(MU[j], SIGMA[j], (1,), generator=rng).item()
    return j, lam


def test_is_reweight_silently_collapses_at_finite_sample_size():
    """The actual T8 finding: draw N real candidates from q_{z_A}. Species 0
    has probability ~1e-7 there, so a run of N=40,000 draws essentially
    never includes it -- the MC estimator silently loses 100% of species
    0's share of the z_B integral, and because the realized sample is
    (accidentally) pure species 1, its OWN reported standard error is tiny:
    confidently wrong, not visibly high-variance.
    """
    N = 40_000
    rng = torch.Generator().manual_seed(99)
    samples = torch.empty(N)
    species0_draws = 0
    for t in range(N):
        j, lam = _sample_species_and_lambda(Z_A, rng)
        if j == 0:
            species0_draws += 1
        q_a = proposal_pdf_fn((j, lam), Z_A)
        samples[t] = target_fn((j, lam), Z_B) / q_a

    assert species0_draws == 0  # the silent part: the rare species never appears at all

    mean = samples.mean().item()
    stderr = samples.std().item() / (N ** 0.5)
    rel_err = (mean - TRUTH_B) / TRUTH_B
    assert rel_err < -0.9  # species 0's entire share of the z_B integral is missing
    assert stderr / TRUTH_B < 0.01  # deceptively tight -- the confidently-wrong signature


if __name__ == "__main__":
    test_mixtures_are_essentially_pure_and_opposite()
    test_naive_near_unbiased_by_cancellation_not_correctness()
    test_is_reweight_exactly_unbiased_in_the_infinite_sample_limit()
    test_is_reweight_silently_collapses_at_finite_sample_size()
    print("all T8 tests passed")
