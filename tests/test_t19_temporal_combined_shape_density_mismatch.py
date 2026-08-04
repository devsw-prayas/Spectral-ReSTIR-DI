"""Point-probe for T19 (session_log_restir_9b Test 3c: combined shape +
density mismatch).

Test 3a/T18 isolated a pure gen/eval AMPLITUDE mismatch on an otherwise
IDENTICAL shape ("same spectral shape both times, isolating the mechanism
from Test 2's shape mismatch"). Test 3c/T19 stacks a genuine SHAPE
distortion on top: the session log's own quadrature-verification section
describes this as history's eval-time target having a different sigma than
current's own target (`sigma_hist=8` vs `sigma_cur=20`) -- i.e. current and
history genuinely have different LOCAL targets at the destination domain,
not just a stale amplitude on an otherwise-shared shape.

`temporal_history.temporal_combine`'s simplified API assumes ONE shared
`target_pdf_fn` for the pair (current and history "share the same
pixel/vertex domain... no distinct per-reservoir target" -- module 7's own
docstring) -- that's exactly right for T17-T18's pure-amplitude case, but
can't express a genuine per-reservoir shape difference. This probe drops
down to `mis_combine.combine_reservoirs` directly (module 4, the same
general multi-target machinery every spatial-reuse T-item already uses) with
TWO distinct target functions, and applies the Coverage Lemma's
`wsum_gen<=gate` drop-history-entirely rule by hand -- exactly what
`temporal_combine` does internally, just made explicit here since the
single-target wrapper doesn't fit this scenario.

Destination = current's own domain (index 0), so the quantity being
estimated is always `integral(A_CUR*L_e*G_CUR)`, independent of history's
own g_gen/g_eval sweep -- unlike T18, TRUTH here is a fixed constant.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from ris_reservoir import Reservoir
from mis_combine import combine_reservoirs

from _temporal_reuse_common import (
    light_spectrum,
    gaussian,
    candidate_mass,
    gen_density,
    quadrature_truth,
)

torch.set_default_dtype(torch.float64)

L_E = light_spectrum()
A_CUR = gaussian(550.0, 70.0)
A_HIST = gaussian(550.0, 25.0)  # narrower band -- genuine shape distortion, same center
G_CUR = 1.3

MASS = candidate_mass(L_E)
P_GEN = gen_density(L_E)
M = 8

TRUTH = quadrature_truth(A_CUR * L_E * G_CUR)


def _target(a, g):
    return lambda idx, a=a, g=g: (a[idx] * L_E[idx] * g).item()


TARGET_CUR = _target(A_CUR, G_CUR)


def _identity_shift(y):
    return y, 1.0


def _stream_reservoir(target_fn, rng):
    r = Reservoir()
    for _ in range(M):
        idx = torch.multinomial(MASS, 1, generator=rng).item()
        p_hat = target_fn(idx)
        p_gen = P_GEN[idx].item()
        accepted = r.update(idx, p_hat / p_gen, rng)
        if accepted:
            r.set_p_hat_gen(p_hat)
    return r


def _run(g_gen, g_eval, N, seed, apply_gate=True):
    rng = torch.Generator().manual_seed(seed)
    target_hist_eval = _target(A_HIST, g_eval)
    target_hist_gen = _target(A_HIST, g_gen)
    target_fns_eval = [TARGET_CUR, target_hist_eval]
    shift_fns = [[_identity_shift, _identity_shift], [_identity_shift, _identity_shift]]

    samples = torch.empty(N)
    for t in range(N):
        r_cur = _stream_reservoir(TARGET_CUR, rng)
        r_hist = _stream_reservoir(target_hist_gen, rng)

        if apply_gate and r_hist.wsum <= 0.0:
            r_hist = Reservoir()  # Coverage Lemma gate: drop entirely, M=0

        combined = combine_reservoirs([r_cur, r_hist], target_fns_eval, shift_fns, dest_index=0, rng=rng)
        samples[t] = combined.wsum
    return samples


def test_shapes_are_genuinely_different():
    assert not torch.allclose(A_CUR, A_HIST)


def test_baseline_nonzero_gen_is_unbiased_despite_shape_mismatch():
    """g_gen=g_eval=1.3: history's shape differs from current's, but its
    gen-time weight has full support (Coverage Lemma holds trivially), so
    the plain MIS combine alone -- no gate needed -- stays unbiased, same
    "shape mismatch alone doesn't need a gate" finding as T2/T4's surface
    spatial-reuse probes."""
    N = 20_000
    samples = _run(g_gen=1.3, g_eval=1.3, N=N, seed=1901, apply_gate=False)
    mean = samples.mean().item()
    stderr = samples.std().item() / (N ** 0.5)
    z = (mean - TRUTH) / stderr
    assert abs(z) < 3.5


def test_disocclusion_without_gate_is_biased_with_a_different_magnitude_than_t18():
    N = 20_000
    samples = _run(g_gen=0.0, g_eval=1.3, N=N, seed=1902, apply_gate=False)
    mean = samples.mean().item()
    rel_err = (mean - TRUTH) / TRUTH
    assert rel_err < -0.05  # decisively wrong -- same hard-break-at-zero mechanism as T18
    # shape mismatch changes the SIZE of the uncorrected bias vs. T18's pure-amplitude
    # ~-50% figure (the balance-heuristic denominator now weighs history's contribution
    # by its own, different-shaped p_hat, not just its raw M-share) -- confirms the
    # Coverage Lemma's bias depends on support, not on shape similarity or magnitude.
    assert abs(rel_err - (-0.5)) > 0.05


def test_disocclusion_with_gate_is_unbiased():
    N = 20_000
    samples = _run(g_gen=0.0, g_eval=1.3, N=N, seed=1902, apply_gate=True)
    mean = samples.mean().item()
    stderr = samples.std().item() / (N ** 0.5)
    z = (mean - TRUTH) / stderr
    assert abs(z) < 3.5


if __name__ == "__main__":
    test_shapes_are_genuinely_different()
    test_baseline_nonzero_gen_is_unbiased_despite_shape_mismatch()
    test_disocclusion_without_gate_is_biased_with_a_different_magnitude_than_t18()
    test_disocclusion_with_gate_is_unbiased()
    print("all T19 tests passed")
