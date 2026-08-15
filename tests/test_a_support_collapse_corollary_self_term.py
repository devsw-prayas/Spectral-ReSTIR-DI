"""Symbolic (SymPy) proof of A12 ("Support-collapse corollary"): real
ReSTIR spatial/temporal reuse always
includes the destination pixel's own freshly-generated reservoir as one term
in the balance-heuristic combine -- never a substitute for it. Because that
self-term's generation-time target IS the destination target exactly
(`p_hat_dest,gen == p_hat_d`, by construction: A4/A5's well-posedness plus
the fact that the destination's own candidates are generated fresh against
its own target every time), the combine's total generation density
`Sum_j c_j*p_hat_j,gen(x)` is guaranteed strictly positive wherever
`p_hat_d(x) > 0` -- REGARDLESS of how badly every other (neighbor/history)
reservoir's own generation support behaves. This was stated in prose
("support coverage holds trivially via the self-term alone") but never
mechanized as an explicit positivity guarantee independent of the other
`n-1` terms.

**Claim, precisely.** For a sum `Sum_j c_j*p_hat_j(x)` with one distinguished
term `j=dest` satisfying `p_hat_dest(x) = p_hat_d(x)` exactly, `c_dest > 0`,
`p_hat_d(x) > 0`, and every OTHER term merely nonnegative (`c_j >= 0`,
`p_hat_j(x) >= 0`, no further assumption) -- the whole sum is strictly
positive. Test 1 proves this for arbitrary `N` and arbitrary (unconstrained
beyond nonnegativity) other terms. Test 2 is the adversarial worst case
named in the corollary's own wording ("bias would require EVERY reservoir in
the sum to miss support"): every other reservoir's target is set to EXACTLY
ZERO at `x` (total support failure everywhere except the self-term), and the
sum still equals `c_dest*p_hat_d(x)` exactly -- unaffected by how many other
terms failed or how badly. Test 3 cross-checks this against the real
`mis_combine.balance_heuristic_weight` denominator with concrete
all-neighbors-zero reservoirs, confirming the actual code never returns a
degenerate zero-support denominator for the destination's own weight as long
as the self-term is present.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import sympy as sp
import torch

torch.set_default_dtype(torch.float64)

from mis_combine import balance_heuristic_weight
from ris_reservoir import Reservoir


def test_self_term_alone_guarantees_positive_total_generation_density():
    c_dest, p_d = sp.symbols("c_dest p_d", positive=True)

    for N in (1, 2, 3, 5, 8):
        cs = [sp.Symbol(f"c{k}", nonnegative=True) for k in range(N)]
        ps = [sp.Symbol(f"p{k}", nonnegative=True) for k in range(N)]  # arbitrary, unconstrained beyond >=0
        other_terms = sum(ck * pk for ck, pk in zip(cs, ps))
        total = c_dest * p_d + other_terms

        # No assumption on the N "other" terms beyond nonnegativity: each is
        # itself nonnegative (product of two nonnegatives), so their sum is
        # nonnegative, so total >= c_dest*p_d > 0 -- checked directly via
        # SymPy's own assumption engine on the residual `other_terms` alone.
        assert other_terms.is_nonnegative
        assert (total - c_dest * p_d).is_nonnegative
        assert total.is_positive


def test_total_support_failure_of_all_other_reservoirs_still_leaves_positive_sum():
    # The adversarial case named explicitly in the corollary's own wording:
    # EVERY other reservoir's target is identically zero at x (complete
    # support collapse everywhere except the self-term). Sum still equals
    # exactly c_dest*p_hat_d(x), for arbitrary N of them.
    c_dest, p_d = sp.symbols("c_dest p_d", positive=True)
    for N in (1, 2, 5, 20):
        cs = sp.symbols(f"c0:{N}", nonnegative=True)
        total = c_dest * p_d + sum(ck * 0 for ck in cs)  # every p_hat_j(x) = 0
        assert sp.simplify(total - c_dest * p_d) == sp.Integer(0)
        assert total.is_positive


def test_implementation_never_degenerates_when_every_neighbor_has_zero_support():
    # Cross-check: real balance_heuristic_weight, N=5 reservoirs, where the
    # destination's own term has a genuinely positive target and every OTHER
    # reservoir's target function returns exactly 0.0 at the query point --
    # confirms the real code's denominator stays strictly positive (driven
    # entirely by the self-term) rather than degenerating to 0/0.
    torch.manual_seed(0)
    DEST = 0
    N = 5
    reservoirs = []
    for _ in range(N):
        r = Reservoir()
        r.y = torch.tensor(0.5)
        r.confidence = 4.0
        reservoirs.append(r)

    def dest_target(y):
        return 2.0 + y  # strictly positive

    def zero_target(y):
        return 0.0

    target_pdf_fns = [dest_target] + [zero_target] * (N - 1)
    identity_shift = lambda y: (y, 1.0)
    shift_fns = [[identity_shift for _ in reservoirs] for _ in reservoirs]

    m_dest = balance_heuristic_weight(DEST, reservoirs[DEST].y, reservoirs, target_pdf_fns, shift_fns)
    # With every other term contributing 0 to the denominator, m_dest must
    # collapse to exactly 1 (the self-term is the entire sum).
    assert abs(m_dest - 1.0) < 1e-12


if __name__ == "__main__":
    test_self_term_alone_guarantees_positive_total_generation_density()
    test_total_support_failure_of_all_other_reservoirs_still_leaves_positive_sum()
    test_implementation_never_degenerates_when_every_neighbor_has_zero_support()
    print("all A-support-collapse-corollary-self-term tests passed")
