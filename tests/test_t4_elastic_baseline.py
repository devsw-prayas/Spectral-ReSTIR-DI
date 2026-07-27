"""Point-probe for T4 (session_log_restir_7 "Test 2": elastic-only vertex,
Tier-0 baseline sanity check).

Same 5-pixel neighborhood as T2, but purely elastic (`K_e` diagonal -- no
fluorescence at all): per-pixel reflectance curves `R_i(lambda)` with mild,
broad, heavily-overlapping variation (a realistic colored-material case,
explicitly NOT a mismatch stress test -- that's T3). Target
`p_hat_i(lambda') = R_i(lambda')*L_e(lambda')*G_i`, light-importance-only
NEE candidate generation, identity shift (elastic vertex type in
`shift_maps.py` is trivially `T==id, J==1`, same as fluorescent).

Historical result: clean pass, z=-1.50, variance reduction only ~1.4x (vs.
T2's ~5x) -- explicitly noted as EXPECTED, not a red flag, because of the
neighborhood's high target overlap (0.99+ coefficient). This probe checks
unbiasedness and that reuse still helps somewhat, but is not held to T2's
bar -- a small reduction here is the correct outcome, not a weak result.
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

# Broad, heavily-overlapping per-pixel reflectance curves -- distinct peaks
# spread across the neighborhood but wide enough (sigma=60) that every
# pixel's target still overlaps substantially with every other's.
R_MU = torch.tensor([450.0, 500.0, 550.0, 600.0, 650.0])
R = torch.stack([gaussian(mu, 60.0) for mu in R_MU])  # (5, N)

MASS = candidate_mass(L_E)
P_GEN = gen_density(L_E)


def _target_pdf(i):
    return lambda idx, i=i: (R[i, idx] * L_E[idx] * G[i]).item()


TARGET_FNS = [_target_pdf(i) for i in range(N_PIXELS)]
SHIFT_FNS = identity_shift_rows()

TRUTH = quadrature_truth(R[DEST] * L_E * G[DEST])


def _draw_reservoir(i, rng):
    idx = torch.multinomial(MASS, 1, generator=rng).item()
    w = TARGET_FNS[i](idx) / P_GEN[idx].item()
    r = Reservoir()
    r.update(idx, w, rng)
    r.set_p_hat_gen(TARGET_FNS[i](idx))
    return r


def test_truth_matches_quadrature_of_the_analytic_target():
    expected = (R[DEST] * L_E * G[DEST] * GRID.weights).sum().item()
    assert abs(TRUTH - expected) < 1e-9


def test_combine_is_unbiased_with_modest_variance_reduction():
    N = 40_000
    rng = torch.Generator().manual_seed(202)

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

    variance_reduction = baseline_samples.var().item() / combined_samples.var().item()
    # Expected to be modest, not dramatic -- high target overlap means
    # reuse helps only a little, and that's the correct/expected outcome
    # here (not a red flag the way it would be for T2's well-separated case).
    assert 1.0 < variance_reduction < 3.0


if __name__ == "__main__":
    test_truth_matches_quadrature_of_the_analytic_target()
    test_combine_is_unbiased_with_modest_variance_reduction()
    print("all T4 tests passed")
