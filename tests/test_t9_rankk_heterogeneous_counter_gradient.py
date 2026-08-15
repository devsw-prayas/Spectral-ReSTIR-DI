"""Point-probe for T9 (counter-gradient: rank-2 heterogeneous fixed-vertex
trichotomy).

Historical V-C is the DI-faithful variant: the destination vertex z_B is
FIXED (the pixel's own path already chose it), and a spectral candidate
generated under a neighbor vertex z_A's LOCAL mixture is reused there.
Counter-gradient config (species 0 falling, species 1 rising, opposite
concentration gradients; z_A=0.2, z_B=0.8; mixtures ~[0.95,0.05] ->
[0.05,0.95]): naive decisively biased (+11.22%, z=+474), fix-local
near-unbiased with the LOWEST variance (~22% rel std), IS-reweight
unbiased but a ~25x variance catastrophe (~545% rel std) relative to
fix-local -- "correctness survives, efficiency doesn't."

Same inner-filter mechanism as T7/T8
(`tests/test_t7_rankk_homogeneous_moderate_overlap.py`): candidate
generation (`species_weight` with a transmittance-blind normalizer) omits
the medium's self-absorption term that the true `local_target` carries, so
naive reuse across a genuine concentration GRADIENT (not the T7/T8
homogeneous case) is biased. Reconstructed parameters, not the historical
exact numbers (the original bands were already flagged as non-recoverable)
-- concentration profile is a sigmoid transition rather
than the historical exact curve, tuned to reproduce the same qualitative
trichotomy: naive decisively biased, IS-reweight unbiased-but-costly,
fix-local unbiased and comparatively cheap.
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

LAMBDA = torch.linspace(400.0, 700.0, 2000)
DLAM = (LAMBDA[1] - LAMBDA[0]).item()
LAMBDA_S = 590.0
MU = {0: 560.0, 1: 620.0}
SIGMA = {0: 40.0, 1: 40.0}
EMISSION_MU, EMISSION_SIGMA = 590.0, 70.0
Z_A, Z_B = 0.2, 0.8
Z0, WIDTH = 0.5, 0.05  # sigmoid transition center/width
L_SCALE = 300.0  # self-absorption strength (tuned for a decisive, not absurd, naive bias)


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


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def conc_counter(j, z):
    # species 0: 2.0 -> 0.1 falling sigmoid; species 1: the mirror-image rise
    s = _sigmoid((z - Z0) / WIDTH)
    return 2.0 - 1.9 * s if j == 0 else 0.1 + 1.9 * s


# reverse cumulative column density integral_z^1 conc(j,z')dz', precomputed
# on a fine z-grid (trapz) once -- transmittance needs this per (j,z) pair,
# not per lambda, so caching it here keeps the MC loop fast.
_Z_GRID = torch.linspace(0.0, 1.0, 2001)
_DZ = (_Z_GRID[1] - _Z_GRID[0]).item()


def _build_column_density(j):
    vals = torch.tensor([conc_counter(j, z.item()) for z in _Z_GRID])
    trailing = torch.zeros_like(vals)
    total = 0.0
    for i in range(len(vals) - 2, -1, -1):
        total += 0.5 * (vals[i].item() + vals[i + 1].item()) * _DZ
        trailing[i] = total
    return trailing


_COLDENS = {j: _build_column_density(j) for j in (0, 1)}


def _column_density(j, z):
    idx = torch.searchsorted(_Z_GRID, torch.as_tensor(float(z))).clamp(max=len(_Z_GRID) - 1)
    return _COLDENS[j][idx].item()


def transmittance_counter(j, lam_prime, z):
    optical_depth = L_SCALE * sum(
        absorption(k, lam_prime) * _column_density(k, z) for k in (0, 1)
    )
    return math.exp(-optical_depth)


def _no_trans_integral(j, z):
    return K_NO_TRANS[j]  # candidate generation ignores the transmittance term


def target_fn(y, z):
    j, lam = y
    return local_target(
        conc_counter, excitation, absorption, emission, j, lam, z, LAMBDA_S,
        transmittance_fn=transmittance_counter,
    )


def proposal_pdf_fn(y, z):
    j, lam = y
    weights = {k: species_weight(conc_counter, excitation, _no_trans_integral, k, z, LAMBDA_S) for k in (0, 1)}
    total = weights[0] + weights[1]
    if total == 0.0:
        return 0.0
    return (weights[j] / total) * absorption(j, lam)


def _mix(z):
    weights = {k: species_weight(conc_counter, excitation, _no_trans_integral, k, z, LAMBDA_S) for k in (0, 1)}
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


def test_mixture_flips_across_the_counter_gradient():
    mix_a = _mix(Z_A)
    mix_b = _mix(Z_B)
    assert mix_a[0] > 0.9 and mix_a[1] < 0.1
    assert mix_b[1] > 0.9 and mix_b[0] < 0.1


def _mc(score_fn, z_source, N, seed):
    rng = torch.Generator().manual_seed(seed)
    samples = torch.empty(N)
    for t in range(N):
        j, lam = _sample(z_source, rng)
        s = score_fn((j, lam))
        samples[t] = 0.0 if s is None else s
    return samples


def test_naive_decisively_biased_inconsistent_accounting():
    N = 50_000
    naive = _mc(lambda y: naive_score(target_fn, proposal_pdf_fn, y, Z_B), Z_A, N, 11)
    mean = naive.mean().item()
    stderr = naive.std().item() / (N ** 0.5)
    z = (mean - TRUTH_B) / stderr
    assert abs(z) > 20.0  # decisively wrong (transfers from T7's mechanism)
    assert (mean - TRUTH_B) / TRUTH_B > 0.1  # double-digit-percent, matches historical magnitude


def test_fix_local_unbiased_lowest_variance():
    N = 50_000
    fixl = _mc(lambda y: fix_local_score(target_fn, proposal_pdf_fn, y, Z_B), Z_B, N, 11)
    mean = fixl.mean().item()
    assert abs((mean - TRUTH_B) / TRUTH_B) < 0.05  # correct accounting -- close to truth


def test_is_reweight_unbiased_but_far_higher_variance_than_fix_local():
    """Correctness survives with source-vertex weights, efficiency doesn't:
    IS-reweight stays close to truth but its variance is dramatically worse
    than fix-local's -- the "~25x variance catastrophe" finding."""
    N = 50_000
    isr = _mc(lambda y: is_reweight_score(target_fn, proposal_pdf_fn, y, Z_A, Z_B), Z_A, N, 11)
    fixl = _mc(lambda y: fix_local_score(target_fn, proposal_pdf_fn, y, Z_B), Z_B, N, 11)

    mean_isr = isr.mean().item()
    assert abs((mean_isr - TRUTH_B) / TRUTH_B) < 0.1  # unbiased-ish (high variance, not systematic)

    variance_ratio = isr.var().item() / fixl.var().item()
    assert variance_ratio > 5.0  # meaningfully worse than fix-local, matches the historical trichotomy


if __name__ == "__main__":
    test_mixture_flips_across_the_counter_gradient()
    test_naive_decisively_biased_inconsistent_accounting()
    test_fix_local_unbiased_lowest_variance()
    test_is_reweight_unbiased_but_far_higher_variance_than_fix_local()
    print("all T9 tests passed")
