"""Point-probe for T5 ("Test 3": dispersive hard TIR-cutoff, bug found and
resolved -- the paper's key `mis_combine.py` motivating story).

Same 5-pixel neighborhood as T2/T4, but each pixel now carries a hard
idealized transmission cutoff `T_i(lambda) = 1{lambda > lambda*_i}` (a
deterministic support boundary, not a smooth Fresnel tail), standing in for
a per-pixel dispersive TIR threshold at slightly different incidence
angles. `shift_maps.py`'s dispersive case is out of scope for the actual
shift (per that module's docstring, a genuine Snell reshift needs real
angle/IOR bookkeeping) -- what T5 stresses is the window mismatch itself,
so shifts stay identity here and only the per-pixel target's hard window
differs (this matches the historical Test 3's own framing: the dispersive
element is the cutoff, not a cross-lambda reconnection).

Historical trajectory (see also `mis_combine.py`'s own docstring): a naive
"canonical" combine --
each source's confidence share `c_i/Sum_j c_j` used as its resampling
weight, with NO reweighting by how well each reservoir's own target
actually matches the destination's -- gave z=-170 at N=20,000, confirmed
as a persistent bias (not slow MC convergence -- it never shrank with more
samples). Root cause: `Reservoir`'s "any f works" unbiasedness only holds
when the reservoir's own target has full support; once each pixel's target
carries a distinct hard cutoff, pooling by confidence share alone treats a
reservoir whose window excludes part of the destination's support exactly
like one that doesn't, dragging the pooled estimate away from truth. The
fix (this IS `mis_combine.combine_reservoirs`, module 4) is the proper
multi-sample balance-heuristic MIS: `m_i(y) = c_i*p_hat_i(y) /
Sum_j c_j*p_hat_j(y)`, which has `p_hat_i(y)` as a factor and is therefore
automatically zero wherever reservoir i has no support -- confirmed
unbiased (z=-0.67, +1.85, -0.37 at N=5,000/20,000/80,000 in the original
log).

This probe reproduces the naive-vs-fixed contrast directly. The exact
original scene parameters and the bias's sign are not recoverable (only
the magnitude/persistence and the fix are documented) -- this
reconstruction checks that the naive confidence-share combine (a) matches
`combine_reservoirs` almost exactly on T2's well-matched scene (confirming
it's a faithful "no MIS correction" baseline, not a differently-broken
formula) and (b) is severely, persistently biased under this file's hard
support mismatch, in either direction; module 4's fix stays unbiased on
the identical scene.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from ris_reservoir import Reservoir
from mis_combine import combine_reservoirs

from _spatial_reuse_common import (
    GRID,
    N_PIXELS,
    light_spectrum,
    gaussian,
    candidate_mass,
    gen_density,
    identity_shift_rows,
    quadrature_truth,
)

torch.set_default_dtype(torch.float64)

L_E = light_spectrum()
A = gaussian(550.0, 70.0)  # broad, well-matched absorption, shared shape across all 5 pixels

# Hard per-pixel transmission cutoffs -- DEST (pixel 0) has the smallest
# (most permissive) threshold, so every other pixel's support window is a
# strict subset of the destination's.
LAM_STAR = torch.tensor([500.0, 520.0, 540.0, 560.0, 580.0])
DEST = 0

WINDOWS = (GRID.lam.unsqueeze(0) > LAM_STAR.unsqueeze(1)).double()  # (5, N)

MASS = candidate_mass(L_E)
P_GEN = gen_density(L_E)


def _target_pdf(i):
    return lambda idx, i=i: (A[idx] * L_E[idx] * WINDOWS[i, idx]).item()


TARGET_FNS = [_target_pdf(i) for i in range(N_PIXELS)]
SHIFT_FNS = identity_shift_rows()

TRUTH = quadrature_truth(A * L_E * WINDOWS[DEST])
M = 4


def _stream_pixel(i, rng):
    r = Reservoir()
    for _ in range(M):
        idx = torch.multinomial(MASS, 1, generator=rng).item()
        p_hat = TARGET_FNS[i](idx)
        p_gen = P_GEN[idx].item()
        accepted = r.update(idx, p_hat / p_gen, rng)
        if accepted:
            r.set_p_hat_gen(p_hat)
    return r


def _naive_confidence_share_combine(reservoirs, target_fns, rng, dest_index=DEST):
    """Pre-MIS naive combine: each source's resampling weight is just its
    confidence share `c_i/Sum_j c_j`, times the destination's own target
    evaluated at its candidate -- no balance-heuristic reweighting by how
    well the source's OWN target matches the destination's. Matches
    `combine_reservoirs` almost exactly when every reservoir shares the same
    target shape (T2's well-matched scene -- see
    `test_t5...matches_combine_reservoirs_on_a_well_matched_scene` below),
    but is exactly the mechanism T5 caught: silently biased once each
    reservoir's own target carries a different hard-cutoff support."""
    total_M = sum(r.M for r in reservoirs)
    combined = Reservoir()
    for r in reservoirs:
        if r.y is None or r.M == 0:
            continue
        m_i = r.M / total_M
        w = m_i * target_fns[dest_index](r.y) * r.contribution_weight()
        accepted = combined.update(r.y, w, rng)
        combined.M += r.M - 1
        if accepted:
            combined.set_p_hat_gen(target_fns[dest_index](r.y))
    return combined


def test_naive_confidence_share_combine_matches_correct_on_a_well_matched_scene():
    # Sanity check that the naive formula is a faithful "no MIS correction"
    # baseline, not a differently-broken one: on a scene where every pixel's
    # target is proportional to a shared shape (T2's well-matched case,
    # rebuilt inline here without the hard cutoff), naive confidence-share
    # and the corrected MIS combine should agree closely -- this is exactly
    # what let the historical bug hide for Tests 1/1b/2 before Test 3 caught it.
    uniform_targets = [lambda idx: (A[idx] * L_E[idx]).item() for _ in range(N_PIXELS)]

    def stream(i, rng):
        r = Reservoir()
        for _ in range(M):
            idx = torch.multinomial(MASS, 1, generator=rng).item()
            p_hat = uniform_targets[i](idx)
            accepted = r.update(idx, p_hat / P_GEN[idx].item(), rng)
            if accepted:
                r.set_p_hat_gen(p_hat)
        return r

    truth_matched = quadrature_truth(A * L_E)

    N = 20_000
    rng_naive = torch.Generator().manual_seed(7)
    naive_samples = torch.empty(N)
    for t in range(N):
        reservoirs = [stream(i, rng_naive) for i in range(N_PIXELS)]
        naive_samples[t] = _naive_confidence_share_combine(reservoirs, uniform_targets, rng_naive, dest_index=0).wsum

    rng_correct = torch.Generator().manual_seed(7)
    correct_samples = torch.empty(N)
    identity_rows = identity_shift_rows()
    for t in range(N):
        reservoirs = [stream(i, rng_correct) for i in range(N_PIXELS)]
        correct_samples[t] = combine_reservoirs(
            reservoirs, uniform_targets, identity_rows, dest_index=0, rng=rng_correct
        ).wsum

    naive_mean = naive_samples.mean().item()
    correct_mean = correct_samples.mean().item()
    assert abs(naive_mean - truth_matched) / truth_matched < 0.05
    assert abs(correct_mean - truth_matched) / truth_matched < 0.05


def test_naive_confidence_share_combine_under_hard_support_mismatch_is_severely_biased():
    N = 20_000
    rng = torch.Generator().manual_seed(5)
    samples = torch.empty(N)
    for t in range(N):
        reservoirs = [_stream_pixel(i, rng) for i in range(N_PIXELS)]
        combined = _naive_confidence_share_combine(reservoirs, TARGET_FNS, rng)
        samples[t] = combined.wsum

    mean = samples.mean().item()
    stderr = samples.std().item() / (N ** 0.5)
    z = (mean - TRUTH) / stderr

    # T5's actual finding: a real, persistent bias under hard support
    # mismatch -- not finite-sample noise. The exact sign isn't recoverable
    # from the historical record (only the magnitude/persistence and the
    # fix are documented), so this checks magnitude only.
    assert abs(z) > 10.0


def test_corrected_mis_combine_is_unbiased_under_the_same_mismatch():
    N = 20_000
    rng = torch.Generator().manual_seed(6)
    samples = torch.empty(N)
    for t in range(N):
        reservoirs = [_stream_pixel(i, rng) for i in range(N_PIXELS)]
        combined = combine_reservoirs(reservoirs, TARGET_FNS, SHIFT_FNS, dest_index=DEST, rng=rng)
        samples[t] = combined.wsum

    mean = samples.mean().item()
    stderr = samples.std().item() / (N ** 0.5)
    z = (mean - TRUTH) / stderr

    assert abs(z) < 3.5  # same scene, same mismatch -- module 4's fix stays unbiased


if __name__ == "__main__":
    test_naive_confidence_share_combine_matches_correct_on_a_well_matched_scene()
    test_naive_confidence_share_combine_under_hard_support_mismatch_is_severely_biased()
    test_corrected_mis_combine_is_unbiased_under_the_same_mismatch()
    print("all T5 tests passed")
