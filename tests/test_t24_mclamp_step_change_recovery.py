"""Point-probe for T24 (M-clamping step-change recovery --
addendum_volumetric_temporal_mclamping.md §10, step-change half).

Companion to T23 (`test_t23_mclamp_input_clamp_steady_state.py`, steady-state
half) -- same multi-frame `temporal_combine` chaining harness, same
continuous rank-1 target family, same M/confidence-split precondition
(session_log_restir_14). Target center jumps 2*sigma mid-sequence.

**Reconstructing this probe surfaced a genuinely new, stronger finding than
the addendum's own framing, worth stating explicitly:** this repo's
`temporal_combine` (a shared-eval-target, two-reservoir confidence-weighted
combine) is the exact family T22 proved `E[h(y)*W]=quadrature_truth(h)` for
EXACTLY, independent of a reservoir's own `p_hat_gen`, as long as
`p_hat_gen>0` wherever `h` has support -- and a smooth, full-support Gaussian
target shift never violates that precondition. A first draft of this test
assumed the addendum's own step-change table ("uncapped still shows -4.6%
bias 74 frames after the shift") meant a genuine per-frame MEAN bias and
asserted a hard rel.err threshold right at the shift frame; it failed. A
direct z-test on the same data showed the "bias" was within ~1.4 sigma of
zero given the sample's own (very large) standard deviation -- i.e. exactly
the heavy-tailed-variance-not-bias trap T22's own module-checkpoint memory
already flags. Likely explanation: the addendum predates
session_log_restir_14's M/confidence fix, so its own step-change numbers may
have been produced under the same chaining bug that session found -- not
re-checked here, since the addendum file itself is out of scope to re-derive.

**What this probe actually checks, confirmed empirically before writing the
assertions below:**
(a) both capped and uncapped stay unbiased in expectation immediately after
a step change AND many frames later (z-test, extending unbiasedness through
chaining + non-stationary targets, no rel.err heuristics);
(b) "ghosting" is real but is a VARIANCE/effective-sample-size effect, not a
mean-bias one -- an uncapped chain's post-shift standard deviation stays far
above its own no-shift steady-state standard deviation for many frames
(pre-shift confidence, never discounted, keeps dominating the resampling
weight distribution's shape), while a capped chain's post-shift standard
deviation returns to its own no-shift steady-state value within a handful of
frames. This is a more precise statement of the addendum's own final
conclusion ("clean bias-variance-responsiveness tradeoff, not a correctness
issue") than its own step-change table achieved.

**Recurring-trap note for future T-items:** when a historical session-log
table reports what looks like a decaying "bias" after some event, z-test it
against the sample's own stderr before assuming it's a real per-frame mean
effect -- large stderr from a confidence-dominated heavy-tailed weight
distribution can look exactly like slow bias decay.
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
MASS = candidate_mass(L_E)
P_GEN = gen_density(L_E)
M = 8

SIGMA_A = 70.0
MU_A = 550.0
MU_B = MU_A + 2 * SIGMA_A  # "target center jumps ~2 sigma", addendum's own step size
G = 1.3


def _target_fn(mu):
    a = gaussian(mu, SIGMA_A)

    def f(idx, a=a):
        return (a[idx] * L_E[idx] * G).item()

    return f


TARGET_A = _target_fn(MU_A)
TARGET_B = _target_fn(MU_B)
TRUTH_B = quadrature_truth(gaussian(MU_B, SIGMA_A) * L_E * G)


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


def _run_chain(target_selector, n_frames, rng, m_cap):
    """See T23's identical helper docstring -- chains `temporal_combine`
    across `n_frames`, feeding each frame's output back in as the next
    frame's history."""
    history = Reservoir()
    wsums = []
    for f in range(n_frames):
        tgt = target_selector(f)
        r_cur = _stream_reservoir(tgt, rng)
        combined = temporal_combine(r_cur, history, tgt, rng, m_cap=m_cap)
        wsums.append(combined.wsum)
        history = combined
    return wsums


def test_step_change_stays_unbiased_and_uncapped_variance_ghosts_longer():
    N = 1500
    N_FRAMES = 40
    SHIFT = 15  # frame index the target jumps from A to B
    LATE = N_FRAMES - 1

    def shifted_selector(f):
        return TARGET_A if f < SHIFT else TARGET_B

    def steady_b_selector(f):
        return TARGET_B

    shifted = {}
    steady = {}
    for seed_offset, m_cap in enumerate((None, 20)):
        rng = torch.Generator().manual_seed(2400 + seed_offset)
        mat = torch.empty(N, N_FRAMES)
        for t in range(N):
            wsums = _run_chain(shifted_selector, N_FRAMES, rng, m_cap)
            mat[t] = torch.tensor(wsums)
        shifted[m_cap] = mat

        rng = torch.Generator().manual_seed(2500 + seed_offset)
        mat = torch.empty(N, N_FRAMES)
        for t in range(N):
            wsums = _run_chain(steady_b_selector, N_FRAMES, rng, m_cap)
            mat[t] = torch.tensor(wsums)
        steady[m_cap] = mat

    def _z(m_cap, frame_idx):
        col = shifted[m_cap][:, frame_idx]
        mean = col.mean().item()
        stderr = col.std().item() / (N ** 0.5)
        return (mean - TRUTH_B) / stderr

    # (a) unbiased in expectation, both configs, right at the shift and late
    # -- extends the shared-eval-target unbiasedness result (T22) through
    # multi-frame chaining and a non-stationary (step-changing) target.
    for m_cap in (None, 20):
        assert abs(_z(m_cap, SHIFT)) < 3.5, f"m_cap={m_cap} frame={SHIFT}"
        assert abs(_z(m_cap, LATE)) < 3.5, f"m_cap={m_cap} frame={LATE}"

    # (b) "ghosting" is a variance/ESS effect, not a mean-bias one: compare
    # each config's post-shift std at the late frame against its OWN
    # no-shift steady-state std at the same frame.
    std_shifted_uncapped = shifted[None][:, LATE].std().item()
    std_steady_uncapped = steady[None][:, LATE].std().item()
    std_shifted_capped = shifted[20][:, LATE].std().item()
    std_steady_capped = steady[20][:, LATE].std().item()

    ratio_uncapped = std_shifted_uncapped / std_steady_uncapped
    ratio_capped = std_shifted_capped / std_steady_capped

    # uncapped: still meaningfully elevated over its own quiet baseline this
    # many frames after the shift (pre-shift confidence never discounted).
    assert ratio_uncapped > 3.0
    # capped: back down near its own quiet baseline within the same window.
    assert ratio_capped < 2.0
    assert ratio_capped < ratio_uncapped


if __name__ == "__main__":
    test_step_change_stays_unbiased_and_uncapped_variance_ghosts_longer()
    print("all T24 tests passed")
