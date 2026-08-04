"""Point-probe for T18 (session_log_restir_9b Test 3a: generation/evaluation
target mismatch -- the real disocclusion mechanism).

Same continuous rank-1 target family as T17, but the history reservoir's
weight was built under `p_hat_hist,gen = G_gen*a(lambda')*L_e(lambda')` (last
frame's scene state) while the combine evaluates
`p_hat_hist,eval = G_eval*a(lambda')*L_e(lambda')` (this frame's) at the same
stored sample -- a pure visibility/brightness factor, SAME spectral shape
both times, isolating the mechanism from T19's shape distortion. Only
`gen=0, eval>0` (disocclusion: history was fully occluded last frame, exposed
this frame) breaks the combine -- and it breaks hard, deterministically,
because the Coverage Lemma's `supp(p_hat_eval) subset supp(p_hat_gen)`
condition fails exactly there. `temporal_history.py`'s `wsum_gen_gate`
(default 0.0) is the derived fix: excluding a `wsum_gen==0` history
reservoir from the combine entirely, not letting its structurally-empty
confidence leak into the MIS denominator.

The reverse (`gen>0, eval=0`) self-corrects with no gate needed -- the
combine's own numerator already zeroes the term out directly, per the
session log's own finding.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from ris_reservoir import Reservoir
from temporal_history import temporal_combine

from _temporal_reuse_common import (
    light_spectrum,
    gaussian,
    candidate_mass,
    gen_density,
    quadrature_truth,
)

torch.set_default_dtype(torch.float64)

L_E = light_spectrum()
A = gaussian(550.0, 70.0)

MASS = candidate_mass(L_E)
P_GEN = gen_density(L_E)
M = 8


def _target(g):
    return lambda idx, g=g: (A[idx] * L_E[idx] * g).item()


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


def _run(g_gen, g_eval, N, seed, wsum_gen_gate=0.0):
    """`g_eval` is THIS FRAME's real target amplitude -- current's own
    gen-time target always matches it trivially (built same frame). `g_gen`
    is what the history reservoir's stored weight was actually built under
    LAST frame -- possibly stale. Both reservoirs are evaluated at the
    destination under the single shared `eval_target` (they share one
    pixel/vertex domain under static geometry), matching
    `temporal_combine`'s own API."""
    rng = torch.Generator().manual_seed(seed)
    eval_target = _target(g_eval)
    samples = torch.empty(N)
    for t in range(N):
        r_cur = _stream_reservoir(eval_target, rng)  # gen ≡ eval trivially, same frame
        r_hist = _stream_reservoir(_target(g_gen), rng)  # stale gen-time weight
        combined = temporal_combine(r_cur, r_hist, eval_target, rng, wsum_gen_gate=wsum_gen_gate)
        samples[t] = combined.wsum
    return samples


def test_baseline_unchanged_gen_eval_is_unbiased():
    N = 20_000
    truth = quadrature_truth(A * L_E * 1.3)
    samples = _run(g_gen=1.3, g_eval=1.3, N=N, seed=1801)
    mean = samples.mean().item()
    stderr = samples.std().item() / (N ** 0.5)
    z = (mean - truth) / stderr
    assert abs(z) < 3.5


def test_disocclusion_without_gate_is_severely_biased():
    """gen=0 (fully occluded last frame), eval>0 (exposed this frame),
    Coverage Lemma condition fails -- disable the gate (permissive
    threshold) to see the raw bias the gate exists to prevent."""
    N = 20_000
    truth = quadrature_truth(A * L_E * 1.3)
    samples = _run(g_gen=0.0, g_eval=1.3, N=N, seed=1802, wsum_gen_gate=-1.0)
    mean = samples.mean().item()
    rel_err = (mean - truth) / truth
    assert rel_err < -0.3  # decisively wrong, matches the session log's ~-50% signature


def test_disocclusion_with_gate_is_unbiased():
    """Same scenario, but temporal_combine's default wsum_gen_gate=0.0 drops
    the structurally-empty history reservoir entirely -- the derived fix."""
    N = 20_000
    truth = quadrature_truth(A * L_E * 1.3)
    samples = _run(g_gen=0.0, g_eval=1.3, N=N, seed=1802)  # default gate
    mean = samples.mean().item()
    stderr = samples.std().item() / (N ** 0.5)
    z = (mean - truth) / stderr
    assert abs(z) < 3.5


def test_reverse_mismatch_self_corrects_without_a_gate():
    """gen>0, eval=0: this frame's real target has vanished entirely
    (truth=0), so BOTH current's own trivial gen≡eval and history's stale
    gen-time weight get evaluated against a zero destination target -- the
    combine's own numerator already zeroes every term out directly, no gate
    needed, per the session log's own finding."""
    N = 20_000
    samples = _run(g_gen=1.3, g_eval=0.0, N=N, seed=1803, wsum_gen_gate=-1.0)
    assert samples.abs().max().item() == 0.0


if __name__ == "__main__":
    test_baseline_unchanged_gen_eval_is_unbiased()
    test_disocclusion_without_gate_is_severely_biased()
    test_disocclusion_with_gate_is_unbiased()
    test_reverse_mismatch_self_corrects_without_a_gate()
    print("all T18 tests passed")
