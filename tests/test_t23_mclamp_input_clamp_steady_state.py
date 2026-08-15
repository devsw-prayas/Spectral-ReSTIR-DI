"""Point-probe for T23 (M-clamping, input-clamp fix vs. broken output-clamp,
steady-state half).

First point-probe to actually *chain* `temporal_combine`'s own output back in
as the next frame's history reservoir across many frames -- every prior
T-item (T2/T6/T12-T22) feeds fresh reservoirs into a combine every trial,
never its own prior output. That untested code path is exactly where a real
bug was once found: `Reservoir.M` was conflating the normalization divisor
`contribution_weight()` needs with the pooled confidence used for MIS,
corrupting readouts the moment a combine's output became a source in a
*later* combine call -- the documented "broken output-clamp" symptom:
clamping the *output* `M_new` after computing `wsum_combine` does not bound
the accumulated weight. This probe only became safe to write once
`ris_reservoir.py`/`mis_combine.py` got the `M`/`confidence` split (see
those modules' docstrings) -- it is the first exercise of that fix, not
just a T-item copy.

**Scoping note on "vs. broken output-clamp":** the broken-output-clamp
failure mode was specifically a single-`M`-field conflation bug, now
eliminated by construction (there is no longer a single field that both
`contribution_weight()`'s divisor and the MIS ratio read, so there is no
longer a meaningful "clamp the wrong field after the fact" mistake left to
reproduce as a parallel code path -- reproducing the pre-fix formula
byte-for-byte here would just re-derive an already-documented finding, not
add new coverage). What this probe actually verifies is the FIXED mechanism
itself: `combine_reservoirs`'s `m_cap` clamps each source's confidence on
the way IN (per `mis_combine.py`'s current docstring), and this stays
unbiased at every cap level once chained across many frames -- the property
that fix was supposed to deliver and the pre-fix version didn't.

Continuous rank-1 target family, same convention as T17-T20
(`p_hat(lambda')=a(lambda')*L_e(lambda')*G`), M=8 candidates/reservoir.

**Steady state** (constant target every frame, cap levels `None/20/100`):
confirms no cap level introduces bias once confidence reaches its capped
steady state -- matching the addendum's own "no cap level introduces
steady-state bias" finding (rel.err within +-1.81 sigma across cap in
{20,40,100} in the historical run).

Step-change recovery speed is T24's own dedicated probe
(`test_t24_mclamp_step_change_recovery.py`), not this file.
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
G = 1.3


def _target_fn(mu):
    a = gaussian(mu, SIGMA_A)

    def f(idx, a=a):
        return (a[idx] * L_E[idx] * G).item()

    return f


TARGET_A = _target_fn(MU_A)
TRUTH_A = quadrature_truth(gaussian(MU_A, SIGMA_A) * L_E * G)


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
    """Chain `temporal_combine` across `n_frames`, feeding each frame's
    combined output back in as the next frame's history reservoir -- the
    exact multi-frame chaining path the M/confidence-conflation bug lived in.
    Returns the per-frame `wsum` list (the unbiased per-frame estimator,
    per `temporal_history.py`'s own established convention)."""
    history = Reservoir()
    wsums = []
    for f in range(n_frames):
        tgt = target_selector(f)
        r_cur = _stream_reservoir(tgt, rng)
        combined = temporal_combine(r_cur, history, tgt, rng, m_cap=m_cap)
        wsums.append(combined.wsum)
        history = combined
    return wsums


def test_steady_state_no_cap_level_introduces_bias():
    N = 2000
    N_FRAMES = 30
    for seed_offset, m_cap in enumerate((None, 20, 100)):
        rng = torch.Generator().manual_seed(2300 + seed_offset)
        samples = torch.empty(N)
        for t in range(N):
            wsums = _run_chain(lambda f: TARGET_A, N_FRAMES, rng, m_cap)
            samples[t] = wsums[-1]  # steady-state per-frame estimate

        mean = samples.mean().item()
        stderr = samples.std().item() / (N ** 0.5)
        z = (mean - TRUTH_A) / stderr
        assert abs(z) < 3.5, f"m_cap={m_cap}: z={z}"


if __name__ == "__main__":
    test_steady_state_no_cap_level_introduces_bias()
    print("all T23 tests passed")
