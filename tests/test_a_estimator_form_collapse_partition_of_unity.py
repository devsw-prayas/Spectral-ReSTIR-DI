"""Symbolic (SymPy) proof of A8's base estimator-form collapse theorem
(`restir_running_notes.md` section 10) at reconnection-valid vertices, for
DISTINCT per-reservoir targets -- a strictly more general claim than
`test_a_balance_heuristic_shared_target_collapse.py`, which additionally
assumed every reservoir shares one common target function. That file's
proof only kicks in downstream of THIS collapse (it starts from the ordinary
balance heuristic already having been reached); this file proves the
upstream step -- substituting A1/A2's `T==id`, `J==1` condition into the
generic shift-aware MIS weight -- and the partition-of-unity property A9's
own Coverage Lemma derivation leans on (`restir_running_notes.md` section 12:
"since Sum_i m_i(x) = 1 wherever p_hat_d(x)>0") but never separately proves.

**Claim 1 (A8's base collapse).** The generic shift-aware weight

    m_i(y) = c_i*p_hat_i(y) / Sum_j [c_j*p_hat_j(T_{i->j}(y))*|dT_{i->j}/dy|]

collapses, once every `T_{i->j}` is substituted with the identity map and
every Jacobian with 1 (A1/A2's reconnection-validity condition, holding
simultaneously across all pairs at such a vertex, per the running notes),
to:

    m_i(y) = c_i*p_hat_i(y) / Sum_j c_j*p_hat_j(y)

for arbitrary DISTINCT per-reservoir target values `p_hat_j(y)` -- direct
substitution, mechanized here for arbitrary N via `sympy.Sum`/`IndexedBase`
(test 1) and for concrete N with fully-expanded sums (test 2), the same
two-track style as the shared-target file.

**Claim 2 (partition of unity).** Summing the collapsed form over `i`:
`Sum_i m_i(y) = Sum_i [c_i*p_hat_i(y)] / Sum_j [c_j*p_hat_j(y)] = 1` whenever
the shared denominator is nonzero -- test 3. This is the identity A9's
Coverage-Lemma derivation invokes to equate `Sum_i m_i(x)*p_hat_d(x)` with
`p_hat_d(x)` itself, and it was never separately mechanized before this file
(the shared-target file's own claim is a further specialization that doesn't
need partition-of-unity explicitly, since collapsing every `p_hat_j` to the
same value makes it trivial by inspection).

Test 4 cross-checks both claims against the real `balance_heuristic_weight`
at concrete values with genuinely DISTINCT per-reservoir targets (not the
shared-target case the other file already covers) and identity shifts.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import sympy as sp
import torch

torch.set_default_dtype(torch.float64)

from mis_combine import balance_heuristic_weight
from ris_reservoir import Reservoir


def test_identity_shift_collapse_symbolic_general_n():
    # Generic shift-aware denominator term: c[j] * p_hat_j(T_{i->j}(y)) *
    # |dT_{i->j}/dy|, kept fully abstract via free symbols Q[j] (the
    # target evaluated at the SHIFTED point) and Jac[j] (the shift's
    # Jacobian) -- i.e. genuinely NOT yet assuming identity shift. A1/A2's
    # reconnection-validity hypothesis is then applied as an explicit
    # substitution (Q[j] -> p[j], the target evaluated at the UNSHIFTED
    # point y, and Jac[j] -> 1), not baked in from the start.
    n, i, j = sp.symbols("n i j", integer=True, positive=True)
    c = sp.IndexedBase("c")
    p = sp.IndexedBase("p")  # p[j] := p_hat_j(y) -- target at the unshifted point
    Q = sp.IndexedBase("Q")  # Q[j] := p_hat_j(T_{i->j}(y)) -- pre-substitution
    Jac = sp.IndexedBase("Jac")  # Jac[j] := |dT_{i->j}/dy| -- pre-substitution

    generic_sum = sp.Sum(c[j] * Q[j] * Jac[j], (j, 1, n))

    # A1/A2's hypothesis: at a reconnection-valid vertex, T_{i->j}=id for
    # EVERY j simultaneously, so Q[j]=p[j] and Jac[j]=1 for every j. Applied
    # to the Sum's summand directly (`.function`), since `j` is bound inside
    # the Sum and a plain top-level `.subs` on the Sum object itself would
    # not reach it.
    collapsed_summand = generic_sum.function.subs({Q[j]: p[j], Jac[j]: 1})
    collapsed_sum = sp.Sum(collapsed_summand, (j, 1, n))

    m_i_collapsed = c[i] * p[i] / collapsed_sum
    m_i_collapsed_explicit = c[i] * p[i] / sp.Sum(c[j] * p[j], (j, 1, n))

    assert sp.simplify(m_i_collapsed - m_i_collapsed_explicit) == sp.Integer(0)


def test_identity_shift_collapse_concrete_n_distinct_targets():
    # Same substitution, fully expanded at concrete N -- here Q_j/Jac_j
    # (pre-shift) and p_j (post-shift) are genuinely separate symbols, and
    # the A1/A2 substitution Q_j->p_j, Jac_j->1 is applied explicitly per
    # term, not assumed from the start.
    for N in (1, 2, 3, 5, 8):
        cs = sp.symbols(f"c0:{N}", positive=True)
        ps = sp.symbols(f"p0:{N}", positive=True)  # DISTINCT per-reservoir values
        Qs = sp.symbols(f"Q0:{N}", positive=True)  # pre-shift target values
        Jacs = sp.symbols(f"Jac0:{N}", positive=True)  # pre-shift Jacobians

        for idx in range(N):
            generic_denom = sum(cj * Qj * Jj for cj, Qj, Jj in zip(cs, Qs, Jacs))
            m_i_generic = cs[idx] * ps[idx] / generic_denom

            # Apply A1/A2: Q_j -> p_j, Jac_j -> 1, for every j simultaneously.
            m_i_collapsed = m_i_generic.subs(
                {**{Qs[k]: ps[k] for k in range(N)}, **{Jacs[k]: 1 for k in range(N)}}
            )
            expected = cs[idx] * ps[idx] / sum(cj * pj for cj, pj in zip(cs, ps))
            assert sp.simplify(m_i_collapsed - expected) == sp.Integer(0)


def test_partition_of_unity_holds_for_distinct_targets():
    # Sum_i m_i(y) == 1 for arbitrary N and arbitrary DISTINCT p_hat_i(y)
    # values -- the identity A9's Coverage Lemma derivation relies on.
    for N in (1, 2, 3, 5, 8):
        cs = sp.symbols(f"c0:{N}", positive=True)
        ps = sp.symbols(f"p0:{N}", positive=True)
        denom = sum(cj * pj for cj, pj in zip(cs, ps))
        total = sum(cs[idx] * ps[idx] / denom for idx in range(N))
        assert sp.simplify(total - 1) == sp.Integer(0)


def test_implementation_matches_derivation_with_distinct_targets_and_identity_shifts():
    # Cross-check against the real balance_heuristic_weight: N=4 reservoirs
    # with GENUINELY DISTINCT target functions (unlike the shared-target
    # file's cross-check), identity shifts throughout -- confirms the actual
    # src/ code reaches the ordinary-balance-heuristic collapse (claim 1)
    # and sums to 1 across reservoirs (claim 2, partition of unity).
    torch.manual_seed(0)
    confidences = [3.0, 7.5, 1.25, 12.0]
    scales = [1.1, 4.0, 0.3, 2.7]  # makes each reservoir's target genuinely distinct
    reservoirs = []
    for c in confidences:
        r = Reservoir()
        r.y = torch.tensor(0.42)
        r.confidence = c
        reservoirs.append(r)

    def make_target(scale):
        return lambda y, scale=scale: scale * (2.5 + 3.0 * y ** 2)

    target_pdf_fns = [make_target(s) for s in scales]
    identity_shift = lambda y: (y, 1.0)
    shift_fns = [[identity_shift for _ in reservoirs] for _ in reservoirs]

    y0 = torch.tensor(0.42)
    p_hat_values = [fn(y0) for fn in target_pdf_fns]
    denom = sum(c * p for c, p in zip(confidences, p_hat_values))

    total = 0.0
    for idx in range(len(reservoirs)):
        m_i = balance_heuristic_weight(idx, reservoirs[idx].y, reservoirs, target_pdf_fns, shift_fns)
        expected = confidences[idx] * p_hat_values[idx] / denom
        assert abs(m_i - expected) < 1e-12
        total += m_i

    assert abs(total - 1.0) < 1e-12  # partition of unity, real code


if __name__ == "__main__":
    test_identity_shift_collapse_symbolic_general_n()
    test_identity_shift_collapse_concrete_n_distinct_targets()
    test_partition_of_unity_holds_for_distinct_targets()
    test_implementation_matches_derivation_with_distinct_targets_and_identity_shifts()
    print("all A-estimator-form-collapse-partition-of-unity tests passed")
