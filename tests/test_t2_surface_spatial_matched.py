"""Point-probe for T2 (session_log_restir_7 "Test 1": fluorescent surface,
well-matched spatial neighborhood).

5-pixel neighborhood, one area light `L_e(lambda)`, UNIFORM fluorophore
`a(lambda)` across all pixels, only a per-pixel geometric factor `G_i`
differs. Target `p_hat_i(lambda') = a(lambda')*L_e(lambda')*G_i`
(`recorrelation_sampler.joint_target`, module 2). Candidate generation:
light-importance-only NEE, `q(lambda') ~ L_e(lambda')`. Combine:
`mis_combine.combine_reservoirs` (module 4) with identity shifts everywhere
-- support is full/matched here, so Tier 1's Jacobian~=1 result and Tier
2's support-coverage corollary both hold trivially (this is exactly the
case `session_log_restir_7` sec 5 notes both the naive and corrected combine
formulas agree on).

Historical result: z=+1.23, ~5x variance reduction vs. no-reuse baseline.
This probe checks the qualitative shape of that result (unbiased, and a
meaningful multi-x variance reduction) rather than the exact historical
z-score, which depends on a since-lost RNG seed/trial count.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from ris_reservoir import Reservoir
from mis_combine import combine_reservoirs

from _spatial_reuse_common import (
    GRID,
    G,
    N_PIXELS,
    DEST,
    light_spectrum,
    gaussian,
    candidate_mass,
    gen_density,
    identity_shift_rows,
    quadrature_truth,
)

torch.set_default_dtype(torch.float64)

L_E = light_spectrum()
A = gaussian(550.0, 70.0)  # uniform fluorophore, same for every pixel, well-matched to the light

MASS = candidate_mass(L_E)
P_GEN = gen_density(L_E)


def _target_pdf(i):
    return lambda idx, i=i: (A[idx] * L_E[idx] * G[i]).item()


TARGET_FNS = [_target_pdf(i) for i in range(N_PIXELS)]
SHIFT_FNS = identity_shift_rows()

TRUTH = quadrature_truth(A * L_E * G[DEST])


def _draw_reservoir(i, rng):
    idx = torch.multinomial(MASS, 1, generator=rng).item()
    w = TARGET_FNS[i](idx) / P_GEN[idx].item()
    r = Reservoir()
    r.update(idx, w, rng)
    r.set_p_hat_gen(TARGET_FNS[i](idx))
    return r


def test_truth_matches_quadrature_of_the_analytic_target():
    expected = (A * L_E * G[DEST] * GRID.weights).sum().item()
    assert abs(TRUTH - expected) < 1e-9


def test_combine_is_unbiased_with_meaningful_variance_reduction():
    N = 40_000
    rng = torch.Generator().manual_seed(101)

    combined_samples = torch.empty(N)
    baseline_samples = torch.empty(N)
    for t in range(N):
        reservoirs = [_draw_reservoir(i, rng) for i in range(N_PIXELS)]
        combined = combine_reservoirs(reservoirs, TARGET_FNS, SHIFT_FNS, dest_index=DEST, rng=rng)
        combined_samples[t] = combined.wsum
        baseline_samples[t] = reservoirs[DEST].wsum

    combined_mean = combined_samples.mean().item()
    combined_stderr = combined_samples.std().item() / (N ** 0.5)
    z_combined = (combined_mean - TRUTH) / combined_stderr
    assert abs(z_combined) < 3.5

    baseline_mean = baseline_samples.mean().item()
    baseline_stderr = baseline_samples.std().item() / (N ** 0.5)
    z_baseline = (baseline_mean - TRUTH) / baseline_stderr
    assert abs(z_baseline) < 3.5  # sanity: no-reuse baseline is also unbiased on its own

    variance_reduction = baseline_samples.var().item() / combined_samples.var().item()
    assert variance_reduction > 2.0  # well-matched neighborhood -- meaningful reuse gain


if __name__ == "__main__":
    test_truth_matches_quadrature_of_the_analytic_target()
    test_combine_is_unbiased_with_meaningful_variance_reduction()
    print("all T2 tests passed")
