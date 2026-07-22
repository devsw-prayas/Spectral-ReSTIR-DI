"""T-tier point-probes for module 7 (temporal_history.py).

Discrete 3-item target (same `p_hat = {0: 1.0, 1: 2.0, 2: 3.0}` family used
by `test_ris_reservoir.py` / `test_mis_combine.py`), covering the Coverage
Lemma (A9) and Temporal Tier-3 Corollary (A10, restir_running_notes.md
§11-12): clean unchanged-history combine (T17), gen/eval target mismatch
that stays unbiased because support never collapses (T18-safe / session
9b's Test 2), the `wsum_gen==0` disocclusion gate dropping a history
reservoir entirely rather than letting its stale confidence pollute the MIS
denominator (T18-unsafe / session 9b's Test 3a), and M-clamping the input
side, not the output (T23).
"""

import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ris_reservoir import Reservoir
from temporal_history import temporal_combine, HistoryBuffer

torch.set_default_dtype(torch.float64)

P_HAT = {0: 1.0, 1: 2.0, 2: 3.0}
P_GEN = 1.0 / 3.0


def target_pdf_fn(y):
    return P_HAT[y]


def _fresh_reservoir(y, target_dict, rng):
    r = Reservoir()
    r.update(y, target_dict[y] / P_GEN, rng)
    r.set_p_hat_gen(target_dict[y])
    return r


def test_clean_unchanged_history_matches_pooled_distribution():
    # T17: current and history both generated under the SAME target this
    # frame (gen ≡ eval trivially). The unbiased per-frame estimator of
    # sum(P_HAT) is `combined.wsum` itself -- exactly module 4's own
    # `F_combined = sum_i m_i(y_i)*p_hat_dest(y_i)*W_i` formula, streamed
    # additively (NOT `combined.contribution_weight()`, which divides by
    # the combined M for downstream chaining across further frames, not
    # per-frame unbiasedness).
    N = 200_000
    rng = torch.Generator().manual_seed(101)
    total = 0.0
    for _ in range(N):
        y_cur = int(torch.randint(0, 3, (1,), generator=rng).item())
        y_hist = int(torch.randint(0, 3, (1,), generator=rng).item())
        r_cur = _fresh_reservoir(y_cur, P_HAT, rng)
        r_hist = _fresh_reservoir(y_hist, P_HAT, rng)
        combined = temporal_combine(r_cur, r_hist, target_pdf_fn, rng)
        total += combined.wsum

    truth = sum(P_HAT.values())
    assert abs(total / N - truth) / truth < 0.01


def test_gen_eval_mismatch_without_support_failure_stays_unbiased():
    # T18-safe / session 9b Test 2: history was generated under a DIFFERENT
    # target (scene changed) that still has full support everywhere the
    # current target does -- Coverage Lemma holds, no gate needed, plain
    # balance-heuristic combine alone stays unbiased.
    target_hist_gen = {0: 4.0, 1: 0.2, 2: 1.5}  # different shape, never zero
    N = 200_000
    rng = torch.Generator().manual_seed(202)
    total = 0.0
    for _ in range(N):
        y_cur = int(torch.randint(0, 3, (1,), generator=rng).item())
        y_hist = int(torch.randint(0, 3, (1,), generator=rng).item())
        r_cur = _fresh_reservoir(y_cur, P_HAT, rng)
        r_hist = _fresh_reservoir(y_hist, target_hist_gen, rng)  # stale gen density
        combined = temporal_combine(r_cur, r_hist, target_pdf_fn, rng)  # eval = P_HAT
        total += combined.wsum

    truth = sum(P_HAT.values())
    assert abs(total / N - truth) / truth < 0.01


def test_wsum_gen_gate_drops_history_entirely():
    # T18-unsafe / session 9b Test 3a: history holds a candidate accumulated
    # over prior frames (M=8, y=0), but this frame's disocclusion means its
    # generation-time target was zero everywhere (wsum==0) -- structurally
    # stale confidence, not just a low-weight candidate.
    rng = torch.Generator().manual_seed(3)
    r_cur = Reservoir()
    r_cur.update(1, P_HAT[1] / P_GEN, rng)
    r_cur.set_p_hat_gen(P_HAT[1])

    r_hist = Reservoir()
    r_hist.y = 0
    r_hist.wsum = 0.0
    r_hist.M = 8
    r_hist.set_p_hat_gen(0.0)

    gated = temporal_combine(r_cur, r_hist, target_pdf_fn, rng)
    assert gated.y == 1
    assert gated.M == r_cur.M  # history contributed zero confidence, not just zero weight

    # Without the gate (wsum_gen_gate below wsum, so it's treated as valid),
    # the stale M=8 still leaks into the combined confidence count even
    # though its own candidate never gets accepted (contribution_weight()==0
    # when p_hat_gen==0) -- the exact "credited share never delivered by
    # anything" mechanism the gate exists to prevent.
    ungated = temporal_combine(r_cur, r_hist, target_pdf_fn, rng, wsum_gen_gate=-1.0)
    assert ungated.M == r_cur.M + r_hist.M
    assert ungated.M != gated.M


def test_m_clamp_input_clamps_history_confidence():
    # T23: history accumulated far more confidence than a capped combine
    # should ever credit it with -- clamp applies to the INPUT M, not the
    # combined output's M after the fact.
    rng = torch.Generator().manual_seed(4)
    r_cur = Reservoir()
    r_cur.update(2, P_HAT[2] / P_GEN, rng)
    r_cur.set_p_hat_gen(P_HAT[2])

    r_hist = Reservoir()
    r_hist.update(1, P_HAT[1] / P_GEN, rng)
    r_hist.set_p_hat_gen(P_HAT[1])
    r_hist.M = 1000  # many frames' worth of accumulated confidence

    capped = temporal_combine(r_cur, r_hist, target_pdf_fn, rng, m_cap=20)
    assert capped.M == r_cur.M + 20

    uncapped = temporal_combine(r_cur, r_hist, target_pdf_fn, rng)
    assert uncapped.M == r_cur.M + 1000


def test_history_buffer_reprojection_is_identity_and_first_frame_is_empty():
    buf = HistoryBuffer()
    unseen = buf.get(pixel_index=42)
    assert unseen.y is None and unseen.M == 0  # no history yet -- gates out cleanly

    r = Reservoir()
    r.update(0, P_HAT[0] / P_GEN, torch.Generator().manual_seed(5))
    r.set_p_hat_gen(P_HAT[0])
    buf.store(pixel_index=7, reservoir=r)

    reprojected = buf.reproject(motion_vectors=object())  # never read, locked static-geometry scope
    assert reprojected is buf
    assert reprojected.get(pixel_index=7) is r


if __name__ == "__main__":
    test_clean_unchanged_history_matches_pooled_distribution()
    test_gen_eval_mismatch_without_support_failure_stays_unbiased()
    test_wsum_gen_gate_drops_history_entirely()
    test_m_clamp_input_clamps_history_confidence()
    test_history_buffer_reprojection_is_identity_and_first_frame_is_empty()
    print("all temporal_history tests passed")
