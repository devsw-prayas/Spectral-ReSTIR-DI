"""Point-probe for T3 (session_log_restir_7 "Test 1b": support-mismatch
stress test + ESS/Rao-Blackwell).

Same 5-pixel neighborhood as T2, but pixel 0 has a narrow, spectrally-offset
absorption tint (sigma=6nm, centered 135nm from the light's peak) instead of
the broad, well-matched `a(lambda)` every other pixel shares -- a real
physical case (textured/tinted pigment), not an occlusion/visibility
effect. Each pixel streams M=4 candidates into its own reservoir before
spatial reuse (a realistic ReSTIR spatial-reuse setup), rather than
T2/T4's single-candidate-per-pixel simplification.

Historical trajectory (`session_log_restir_7_tier4_spatial_reuse_probes.md`
sec 2): first-pass canonical combine looked biased (z=-16.96 at N=20,000);
an ESS diagnostic on pixel 0's reservoir revealed near-total weight
degeneracy (ESS~1.02/4); a Rao-Blackwellized re-check (average pixel 0's
own M candidates directly instead of picking one via its reservoir's
internal accept/reject) converged immediately -- confirmed unbiased, the
z=-17 was pure finite-sample heavy-tailed MC variance, not a real bug.

This probe reproduces the STRUCTURAL lesson deterministically rather than
replaying the exact historical z-scores (those depend on a since-lost RNG
seed and scene scale): it asserts (a) pixel 0's reservoir is genuinely
near-degenerate (ESS/M well below 1), (b) both the plain streamed combine
and a Rao-Blackwellized combine are unbiased in expectation, and (c) the
RB combine has strictly lower variance -- the actual point of
Rao-Blackwellization (law of total variance), not a bias fix.

The Rao-Blackwellization itself isn't exposed by any src/ module (per
`forward_paper1_test_suite.md`'s note that this is a diagnostic technique,
not new theory) -- it's implemented here directly on top of
`mis_combine.balance_heuristic_weight` (module 4) and
`furnace_canary.effective_sample_size` (module 9).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from ris_reservoir import Reservoir
from mis_combine import combine_reservoirs, balance_heuristic_weight
from furnace_canary import effective_sample_size

from _spatial_reuse_common import (
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
A_UNIFORM = gaussian(550.0, 70.0)  # broad, well-matched absorption every pixel but 0 shares
A_NARROW = gaussian(415.0, 6.0)  # pixel 0: narrow, ~135nm offset from the light's peak

MASS = candidate_mass(L_E)
P_GEN = gen_density(L_E)


def _target_pdf(i):
    a = A_NARROW if i == 0 else A_UNIFORM
    return lambda idx, a=a, i=i: (a[idx] * L_E[idx] * G_I[i]).item()


G_I = torch.tensor([1.0, 1.2, 0.8, 1.5, 0.9])
TARGET_FNS = [_target_pdf(i) for i in range(N_PIXELS)]
SHIFT_FNS = identity_shift_rows()

TRUTH = quadrature_truth(A_UNIFORM * L_E * G_I[DEST])
M = 4


def _stream_pixel(i, rng, record=False):
    r = Reservoir()
    draws = []
    for _ in range(M):
        idx = torch.multinomial(MASS, 1, generator=rng).item()
        p_hat = TARGET_FNS[i](idx)
        p_gen = P_GEN[idx].item()
        w = p_hat / p_gen
        accepted = r.update(idx, w, rng)
        if accepted:
            r.set_p_hat_gen(p_hat)
        if record:
            draws.append((idx, w))
    return r, draws


def _rb_pixel0_contribution(draws0, reservoirs):
    """Rao-Blackwellized replacement for pixel 0's term: average the M raw
    candidate draws' own single-candidate MIS contribution directly instead
    of letting the reservoir's internal accept/reject pick just one."""
    total = 0.0
    for idx, _w in draws0:
        m_0 = balance_heuristic_weight(0, idx, reservoirs, TARGET_FNS, SHIFT_FNS)
        y_dest, J = SHIFT_FNS[0][DEST](idx)
        if y_dest is None:
            continue
        total += m_0 * TARGET_FNS[DEST](y_dest) * abs(J) / P_GEN[idx].item()
    return total / len(draws0)


def _combined_wsum(reservoirs, override_index=None, override_value=None):
    total = 0.0
    for i, r in enumerate(reservoirs):
        if i == override_index:
            total += override_value
            continue
        if r.y is None or r.M == 0:
            continue
        m_i = balance_heuristic_weight(i, r.y, reservoirs, TARGET_FNS, SHIFT_FNS)
        y_dest, J = SHIFT_FNS[i][DEST](r.y)
        if y_dest is None:
            continue
        total += m_i * TARGET_FNS[DEST](y_dest) * abs(J) * r.contribution_weight()
    return total


def test_pixel0_reservoir_is_near_degenerate():
    rng = torch.Generator().manual_seed(11)
    _, draws = _stream_pixel(0, rng, record=True)
    weights = torch.tensor([w for _, w in draws])
    ess = effective_sample_size(weights)
    assert ess / M < 0.5  # near-total weight degeneracy signature (session 7: ~0.255)


def test_naive_and_rb_combine_are_both_unbiased_but_rb_has_lower_variance():
    N = 20_000
    rng = torch.Generator().manual_seed(12)
    naive_samples = torch.empty(N)
    rb_samples = torch.empty(N)

    for t in range(N):
        reservoirs = []
        draws0 = None
        for i in range(N_PIXELS):
            r, draws = _stream_pixel(i, rng, record=(i == 0))
            reservoirs.append(r)
            if i == 0:
                draws0 = draws

        naive = combine_reservoirs(reservoirs, TARGET_FNS, SHIFT_FNS, dest_index=DEST, rng=rng)
        naive_samples[t] = naive.wsum

        rb0 = _rb_pixel0_contribution(draws0, reservoirs)
        rb_samples[t] = _combined_wsum(reservoirs, override_index=0, override_value=rb0)

    for samples in (naive_samples, rb_samples):
        mean = samples.mean().item()
        stderr = samples.std().item() / (N ** 0.5)
        z = (mean - TRUTH) / stderr
        assert abs(z) < 3.5

    assert rb_samples.var().item() < naive_samples.var().item()


if __name__ == "__main__":
    test_pixel0_reservoir_is_near_degenerate()
    test_naive_and_rb_combine_are_both_unbiased_but_rb_has_lower_variance()
    print("all T3 tests passed")
