"""Symbolic (SymPy) proof that `mis_combine.balance_heuristic_weight`'s
generalized MIS weight `m_i` collapses exactly to the plain confidence share
`M_i / Sum_j M_j` whenever every reservoir shares one common eval-time target
and every pairwise shift between them is identity (T27's family: T17/T21/T22's
own shared-single-eval-target temporal setup, and T32's spatial analog).

A dedicated MC-adjacent test
(`test_balance_heuristic_collapses_to_confidence_share_under_shared_target`
in `test_t22_volumetric_temporal_cgen_sweep_rb.py`) already confirms this
collapse as an exact equality, but that check exercises the real
`src/mis_combine.py` code at concrete floating-point values, not a proof
that the collapse holds for arbitrary confidences / arbitrary N / an
arbitrary (nonzero) shared target value. This file supplies that proof, then
cross-checks it directly against the actual `balance_heuristic_weight`
implementation at concrete values as an implementation-matches-derivation
sanity anchor (not a re-derivation of the proof itself).

**The algebra** (the generalized MIS weight formula, specialized to identity
shifts J=1 and one shared `target_pdf_fn`, `p_hat_j = p_hat` for every j):

    m_i(y) = c_i * p_hat(y) / Sum_j [c_j * p_hat(y)]
           = c_i * p_hat(y) / [p_hat(y) * Sum_j c_j]
           = c_i / Sum_j c_j                          (whenever p_hat(y) != 0)

The `p_hat(y)` factor is IDENTICAL across every term in the denominator's sum
precisely because every reservoir shares one target function and the shift
maps between them are all identity (so every reservoir's sample lands at the
same evaluation point `y` in every other reservoir's domain, per this repo's
own `shift_maps.py` identity-shift convention) -- it factors out of the sum
entirely and cancels against the numerator's own `p_hat(y)` factor. This is
what makes the result independent of `p_hat`'s actual value/shape: the proof
only needs `p_hat(y) != 0`, never what `p_hat` computes.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import sympy as sp
import torch

torch.set_default_dtype(torch.float64)

from mis_combine import balance_heuristic_weight
from ris_reservoir import Reservoir


def test_collapse_holds_for_arbitrary_n_general_symbolic_proof():
    n, j, i = sp.symbols("n j i", integer=True, positive=True)
    c = sp.IndexedBase("c")
    P = sp.symbols("P", nonzero=True)  # shared p_hat(y) value, arbitrary nonzero

    m_i = c[i] * P / sp.Sum(c[j] * P, (j, 1, n))
    confidence_share = c[i] / sp.Sum(c[j], (j, 1, n))

    residual = sp.simplify(m_i - confidence_share)
    assert residual == sp.Integer(0)


def test_collapse_holds_for_concrete_small_n_fully_expanded():
    # Same claim, re-derived with fully expanded (non-abstract-Sum) symbolic
    # sums at several concrete N -- a second, independent code path through
    # SymPy's simplifier (no `Sum`/`IndexedBase` machinery at all), guarding
    # against a false-positive from the abstract-Sum proof's own machinery.
    for N in (1, 2, 3, 5, 8):
        cs = sp.symbols(f"c0:{N}", positive=True)
        P = sp.symbols("P", nonzero=True)
        denom = sum(cj * P for cj in cs)
        for idx in range(N):
            m_i = cs[idx] * P / denom
            target = cs[idx] / sum(cs)
            assert sp.simplify(m_i - target) == sp.Integer(0)


def test_implementation_matches_derivation_at_concrete_values():
    # Cross-check: the real `balance_heuristic_weight` function, run on N=4
    # reservoirs sharing one target and identity shifts, must numerically
    # match the closed-form `c_i / Sum_j c_j` the proof above derives --
    # confirms the src/ code actually implements the algebra just proved,
    # not just that the algebra itself is internally consistent.
    torch.manual_seed(0)
    confidences = [3.0, 7.5, 1.25, 12.0]
    reservoirs = []
    for c in confidences:
        r = Reservoir()
        r.y = torch.tensor(0.42)
        r.confidence = c
        reservoirs.append(r)

    def shared_target(y):
        return 2.5 + 3.0 * y**2  # arbitrary nonzero shape, value irrelevant to claim

    target_pdf_fns = [shared_target] * len(reservoirs)
    identity_shift = lambda y: (y, 1.0)
    shift_fns = [[identity_shift for _ in reservoirs] for _ in reservoirs]

    total_c = sum(confidences)
    for idx in range(len(reservoirs)):
        m_i = balance_heuristic_weight(idx, reservoirs[idx].y, reservoirs, target_pdf_fns, shift_fns)
        expected = confidences[idx] / total_c
        assert abs(m_i - expected) < 1e-12


if __name__ == "__main__":
    test_collapse_holds_for_arbitrary_n_general_symbolic_proof()
    test_collapse_holds_for_concrete_small_n_fully_expanded()
    test_implementation_matches_derivation_at_concrete_values()
    print("all A-balance-heuristic-shared-target-collapse tests passed")
