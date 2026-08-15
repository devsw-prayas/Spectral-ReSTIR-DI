"""Point-probe for T10 (support failure: rank-2 heterogeneous fixed-vertex,
-78.4%-class hard support violation).

Historical support-failure config: species 0's concentration hits an
exact hard step to zero above z0=0.5. Candidates generated in the dead zone
(z_A=0.8, c1(z_A)=0) are reused at z_B=0.2 where c1(z_B)>0. Result: naive
-18.3% (transfers, expected), fix-local -0.03% (clean, as always), and
IS-reweight **-78.4%** with a deceptively tight 4.8% error bar --
`is_reweight_score`'s structural-bias regime (`has_support_violation`),
not the ordinary high-variance regime T9 exercises. This is the case A6's
trichotomy calls out by name: "structurally biased, not just high-variance"
the moment containment `supp(p_hat(.;z_B)) subset supp(q_{z_A})` fails --
exact-zero concentration regions are *generic* in heterogeneous media
(plumes, tissue boundaries), so this is flagged as the dangerous production
regime, not an edge case.

Unlike T7-T9, this needs no transmittance mechanism to produce bias --
`heterogeneous_lookup.is_reweight_score` already returns `None` (existence
failure) whenever the source-vertex proposal has zero mass for a species
that's alive at the destination (see module docstring and
`test_heterogeneous_lookup.py`'s own `test_is_reweight_structurally_biased_under_support_violation`,
which validates the identical mechanism at exact-quadrature precision).
This probe promotes that same mechanism to a T-tier MC point-probe with the
historical z_A/z_B roles and z-score/variance framing, using freshly
reconstructed parameters (the original historical parameters were never
recorded precisely enough to reproduce exactly).
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
    has_support_violation,
    lookup_trichotomy_case,
)

torch.set_default_dtype(torch.float64)

LAMBDA = torch.linspace(400.0, 700.0, 2000)
DLAM = (LAMBDA[1] - LAMBDA[0]).item()
LAMBDA_S = 590.0
MU = {0: 560.0, 1: 620.0}
SIGMA = {0: 40.0, 1: 40.0}
EMISSION_MU, EMISSION_SIGMA = 590.0, 70.0
Z0 = 0.5  # species 0's hard cutoff: alive (c=1.5) below, exactly dead (c=0) above
Z_A, Z_B = 0.8, 0.2  # candidates generated in the dead zone, reused where species 0 is alive


def _gaussian_pdf(x, mu, sigma):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


A_TENSOR = {j: _gaussian_pdf(LAMBDA, MU[j], SIGMA[j]) for j in (0, 1)}
LE_TENSOR = _gaussian_pdf(LAMBDA, EMISSION_MU, EMISSION_SIGMA)
INTEGRAL_A_LE = {j: torch.trapz(A_TENSOR[j] * LE_TENSOR, LAMBDA).item() for j in (0, 1)}


def absorption(j, lam):
    return _gaussian_pdf(torch.as_tensor(float(lam)), MU[j], SIGMA[j]).item()


def emission(lam):
    return _gaussian_pdf(torch.as_tensor(float(lam)), EMISSION_MU, EMISSION_SIGMA).item()


def excitation(j, lambda_s):
    return 1.0


def conc_cutoff(j, z):
    if j == 0:
        return 1.5 if z < Z0 else 0.0  # exact hard zero above the cutoff
    return 1.0  # species 1 always present


def _integral_a_le(j, z):
    return INTEGRAL_A_LE[j]  # no transmittance term needed for this mechanism


def target_fn(y, z):
    j, lam = y
    return local_target(conc_cutoff, excitation, absorption, emission, j, lam, z, LAMBDA_S)


def proposal_pdf_fn(y, z):
    j, lam = y
    weights = {k: species_weight(conc_cutoff, excitation, _integral_a_le, k, z, LAMBDA_S) for k in (0, 1)}
    total = weights[0] + weights[1]
    if total == 0.0:
        return 0.0
    return (weights[j] / total) * absorption(j, lam)


def _mix(z):
    weights = {k: species_weight(conc_cutoff, excitation, _integral_a_le, k, z, LAMBDA_S) for k in (0, 1)}
    total = weights[0] + weights[1]
    return weights[0] / total, weights[1] / total


def _quadrature(z):
    total = 0.0
    for j in (0, 1):
        for lam in LAMBDA:
            total += target_fn((j, lam.item()), z) * DLAM
    return total


TRUTH_B = _quadrature(Z_B)


def _sample(z, rng):
    w0, _ = _mix(z)
    j = 0 if torch.rand((), generator=rng).item() < w0 else 1
    lam = torch.normal(MU[j], SIGMA[j], (1,), generator=rng).item()
    return j, lam


def _mc(score_fn, z_source, N, seed):
    rng = torch.Generator().manual_seed(seed)
    samples = torch.empty(N)
    for t in range(N):
        j, lam = _sample(z_source, rng)
        s = score_fn((j, lam))
        samples[t] = 0.0 if s is None else s
    return samples


def test_species0_is_dead_at_source_alive_at_destination():
    assert conc_cutoff(0, Z_A) == 0.0
    assert conc_cutoff(0, Z_B) > 0.0
    assert has_support_violation(conc_cutoff, 0, Z_A, Z_B)
    assert not has_support_violation(conc_cutoff, 1, Z_A, Z_B)
    assert lookup_trichotomy_case(conc_cutoff, (0, MU[0]), Z_A, Z_B) == "support_violation"
    assert lookup_trichotomy_case(conc_cutoff, (1, MU[1]), Z_A, Z_B) == "reweight_safe"
    mix_a = _mix(Z_A)
    assert mix_a[0] == 0.0  # species 0 literally never proposed at the source


def test_is_reweight_structurally_biased_with_deceptively_tight_error_bar():
    """The A6-trichotomy case this T-item exists to catch: bias, not
    variance -- `is_reweight_score` returns `None` for every species-0
    candidate scored at z_B (existence failure, dropped per the
    drop-not-zero-fill discipline), so species 0's ENTIRE contribution to
    the z_B integral silently vanishes from the estimator, at a reported
    error bar that looks fine."""
    N = 80_000
    isr = _mc(lambda y: is_reweight_score(target_fn, proposal_pdf_fn, y, Z_A, Z_B), Z_A, N, 22)
    mean = isr.mean().item()
    stderr = isr.std().item() / (N ** 0.5)
    rel_err = (mean - TRUTH_B) / TRUTH_B
    assert rel_err < -0.3  # decisive, double-digit-percent underestimate (historical: -78.4%)
    assert abs((mean - TRUTH_B) / stderr) > 50.0  # deceptively tight error bar, not "just noisy"


def test_naive_and_fix_local_unaffected_by_the_support_boundary():
    N = 80_000
    naive = _mc(lambda y: naive_score(target_fn, proposal_pdf_fn, y, Z_B), Z_A, N, 22)
    fixl = _mc(lambda y: fix_local_score(target_fn, proposal_pdf_fn, y, Z_B), Z_B, N, 22)
    assert abs((naive.mean().item() - TRUTH_B) / TRUTH_B) < 0.1
    assert abs((fixl.mean().item() - TRUTH_B) / TRUTH_B) < 0.1


if __name__ == "__main__":
    test_species0_is_dead_at_source_alive_at_destination()
    test_is_reweight_structurally_biased_with_deceptively_tight_error_bar()
    test_naive_and_fix_local_unaffected_by_the_support_boundary()
    print("all T10 tests passed")
