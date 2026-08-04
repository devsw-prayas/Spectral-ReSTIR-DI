"""Point-probe for T17 (session_log_restir_9b Test 1: clean temporal
baseline, unchanged history).

Continuous rank-1 fluorophore target `p_hat(lambda') = a(lambda')*L_e(lambda')*G`,
current and history reservoirs both generated under the SAME target this
frame (gen ≡ eval trivially for both, static-geometry locked scope --
`temporal_history.py`'s own precondition). M=8 candidates/reservoir,
light-importance proposal `q ~ L_e`, same convention as the session log.
First numeric confirmation that `temporal_combine` (module 7, wired directly
on top of `mis_combine.combine_reservoirs`, module 4) is unbiased in the
trivial unchanged-history case -- the base case the Coverage Lemma's
`iff` condition is automatically satisfied for (gen≡eval, so
`supp(p_hat_eval) subset supp(p_hat_gen)` holds trivially).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from ris_reservoir import Reservoir
from temporal_history import temporal_combine

from _temporal_reuse_common import (
    GRID,
    light_spectrum,
    gaussian,
    candidate_mass,
    gen_density,
    quadrature_truth,
)

torch.set_default_dtype(torch.float64)

L_E = light_spectrum()
A = gaussian(550.0, 70.0)
G = 1.3

MASS = candidate_mass(L_E)
P_GEN = gen_density(L_E)
M = 8


def target_pdf_fn(idx):
    return (A[idx] * L_E[idx] * G).item()


TRUTH = quadrature_truth(A * L_E * G)


def _stream_reservoir(rng):
    r = Reservoir()
    for _ in range(M):
        idx = torch.multinomial(MASS, 1, generator=rng).item()
        p_hat = target_pdf_fn(idx)
        p_gen = P_GEN[idx].item()
        accepted = r.update(idx, p_hat / p_gen, rng)
        if accepted:
            r.set_p_hat_gen(p_hat)
    return r


def test_truth_matches_quadrature_of_the_analytic_target():
    expected = (A * L_E * G * GRID.weights).sum().item()
    assert abs(TRUTH - expected) < 1e-9


def test_clean_unchanged_history_combine_is_unbiased():
    N = 40_000
    rng = torch.Generator().manual_seed(1700)

    combined_samples = torch.empty(N)
    for t in range(N):
        r_cur = _stream_reservoir(rng)
        r_hist = _stream_reservoir(rng)
        combined = temporal_combine(r_cur, r_hist, target_pdf_fn, rng)
        combined_samples[t] = combined.wsum

    mean = combined_samples.mean().item()
    stderr = combined_samples.std().item() / (N ** 0.5)
    z = (mean - TRUTH) / stderr
    assert abs(z) < 3.5


if __name__ == "__main__":
    test_truth_matches_quadrature_of_the_analytic_target()
    test_clean_unchanged_history_combine_is_unbiased()
    print("all T17 tests passed")
