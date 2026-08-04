"""Point-probe for T20 (Session 9's informal adversarial probe: reservoir-
weight bug v1, patched with a hard reject gate v2 -- precursor to A9/A10,
no A-item support of its own).

Per `restir_full_inventory_chronological.md`'s Session 9 entry: "caught a
reservoir-weight formula bug (v1), patched with a hard reject gate (v2) --
without establishing whether the corrected formula alone (no gate) would
already have been unbiased. Left open, closed in Session 9b." The original
Session 9 log was never committed as a standalone file (session_log_restir_9b's
own header says so explicitly), so this probe reconstructs the STRUCTURE of
that finding from the inventory's description, using this repo's own
established "naive confidence-share" combine bug (the exact mechanism T5
caught in the spatial case: `m_i = c_i / sum_j c_j`, no target-shape
reweighting at all) as bug v1 -- the same failure class the session log
calls "reproducing the known spatial failure mode temporally."

Three things this probe checks, matching the historical narrative precisely:
1. v1 (naive confidence-share) is severely biased under a full-disocclusion
   scenario (history has M>0 stale confidence but zero real gen-time mass).
2. v2 (v1 patched with a hard reject-if-wsum-below-tau gate) fixes THAT
   specific tested case.
3. **v2 is a blunt instrument, unlike the later-derived exact rule**: applied
   with the SAME conservative tau to a genuinely SAFE case (tiny but nonzero
   gen-time density -- Coverage-Lemma-safe, per session_log_restir_9b Test
   3b's own finding "any nonzero G_gen, however tiny, is fully clean"), v2's
   graduated threshold discards history almost every trial anyway, losing
   the temporal-reuse variance benefit for no bias reason -- while
   `temporal_history.temporal_combine`'s actual derived rule
   (`wsum_gen_gate=0.0`, an EXACT zero-vs-nonzero cliff, not a graduated
   proxy) keeps this safe case's history and gets a real variance reduction
   for it. This is the concrete content of "left open, closed in Session
   9b" -- v2 could not have known the exact cliff without the Coverage
   Lemma's derivation.
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
G_EVAL = 1.3

MASS = candidate_mass(L_E)
P_GEN = gen_density(L_E)
M = 8

TRUTH = quadrature_truth(A * L_E * G_EVAL)


def _target(g):
    return lambda idx, g=g: (A[idx] * L_E[idx] * g).item()


EVAL_TARGET = _target(G_EVAL)


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


def naive_v1_combine(r_cur, r_hist, target_pdf_fn):
    """Bug v1: confidence-share-only weighting (T5's exact naive mechanism,
    reused here temporally), no target-shape reweighting in the denominator
    at all -- credits history a `c_hist/(c_cur+c_hist)` share regardless of
    whether it can deliver anything."""
    c_cur, c_hist = r_cur.M, r_hist.M
    total = c_cur + c_hist
    if total == 0:
        return 0.0
    value = 0.0
    if r_cur.y is not None:
        value += (c_cur / total) * target_pdf_fn(r_cur.y) * r_cur.contribution_weight()
    if r_hist.y is not None:
        value += (c_hist / total) * target_pdf_fn(r_hist.y) * r_hist.contribution_weight()
    return value


TAU = 100.0  # graduated proxy threshold, Session 9-style -- not derived from the Coverage Lemma


def gated_v2_combine(r_cur, r_hist, target_pdf_fn, tau=TAU):
    """Gate v2: hard reject the whole history reservoir if its realized
    wsum falls below a graduated proxy threshold `tau` (the kind of
    ad hoc cutoff Session 9 assumed before the Coverage Lemma derived the
    EXACT wsum_gen==0 rule)."""
    if r_hist.wsum < tau:
        r_hist = Reservoir()
    return naive_v1_combine(r_cur, r_hist, target_pdf_fn)


def _run_naive(g_gen, N, seed, gate_fn=None):
    rng = torch.Generator().manual_seed(seed)
    hist_target = _target(g_gen)
    samples = torch.empty(N)
    m_hist_retained = torch.empty(N)
    for t in range(N):
        r_cur = _stream_reservoir(EVAL_TARGET, rng)
        r_hist = _stream_reservoir(hist_target, rng)
        if gate_fn is None:
            samples[t] = naive_v1_combine(r_cur, r_hist, EVAL_TARGET)
            m_hist_retained[t] = r_hist.M
        else:
            eff_hist = gate_fn(r_hist)
            samples[t] = naive_v1_combine(r_cur, eff_hist, EVAL_TARGET)
            m_hist_retained[t] = eff_hist.M
    return samples, m_hist_retained


def test_bug_v1_naive_confidence_share_is_severely_biased_under_disocclusion():
    N = 20_000
    samples, _ = _run_naive(g_gen=0.0, N=N, seed=2001)
    mean = samples.mean().item()
    rel_err = (mean - TRUTH) / TRUTH
    assert rel_err < -0.3  # decisively wrong, same class of failure as T5/T18


def test_gate_v2_fixes_the_tested_disocclusion_case():
    N = 20_000
    rng = torch.Generator().manual_seed(2002)
    samples = torch.empty(N)
    for t in range(N):
        r_cur = _stream_reservoir(EVAL_TARGET, rng)
        r_hist = _stream_reservoir(_target(0.0), rng)
        samples[t] = gated_v2_combine(r_cur, r_hist, EVAL_TARGET)
    mean = samples.mean().item()
    stderr = samples.std().item() / (N ** 0.5)
    z = (mean - TRUTH) / stderr
    assert abs(z) < 3.5


def test_gate_v2_is_a_blunt_instrument_on_a_safe_small_case():
    """g_gen=0.02: tiny but nonzero -- Coverage-Lemma-safe (T18's own
    reverse/baseline logic: any nonzero gen density is fully clean, no gate
    needed). v2's graduated tau still discards history almost every trial
    here (mean M_hist retained near 0), unlike the derived exact rule
    (wsum_gen_gate=0.0), which only drops a reservoir whose wsum is
    EXACTLY zero -- this safe case's wsum is never exactly zero, so the
    derived rule keeps it every trial."""
    N = 20_000
    g_gen = 0.02

    _, m_hist_v2 = _run_naive(g_gen=g_gen, N=N, seed=2003, gate_fn=lambda r: (Reservoir() if r.wsum < TAU else r))
    assert m_hist_v2.mean().item() < M * 0.5  # v2 discards history most of the time here

    rng = torch.Generator().manual_seed(2003)
    m_hist_derived = torch.empty(N)
    for t in range(N):
        r_cur = _stream_reservoir(EVAL_TARGET, rng)
        r_hist = _stream_reservoir(_target(g_gen), rng)
        combined = temporal_combine(r_cur, r_hist, EVAL_TARGET, rng)  # default wsum_gen_gate=0.0
        m_hist_derived[t] = M if combined.M > r_cur.M else 0  # history contributed real confidence?

    assert m_hist_derived.mean().item() > m_hist_v2.mean().item() + M * 0.3  # derived rule keeps it far more often


if __name__ == "__main__":
    test_bug_v1_naive_confidence_share_is_severely_biased_under_disocclusion()
    test_gate_v2_fixes_the_tested_disocclusion_case()
    test_gate_v2_is_a_blunt_instrument_on_a_safe_small_case()
    print("all T20 tests passed")
