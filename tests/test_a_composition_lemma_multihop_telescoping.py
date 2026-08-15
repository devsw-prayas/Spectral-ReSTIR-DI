"""Symbolic (SymPy) proof of T29's multi-hop chaining claim (composition
lemma: does the single-hop reprojection-Jacobian identity survive being
chained across many hops?).
`test_t29_multihop_chaining_stability.py` confirms only that a T-hop
SEQUENTIALLY-CHAINED Jacobian-ratio product matches a DIRECT single-shot
ratio at machine precision (~1e-15), for concrete T in {10,...,500} and
concrete floating-point deformation fields -- a numerical-robustness check,
not a proof that chaining is correct in exact arithmetic for arbitrary T.

**The claim to mechanize.** T27's own proof
(`test_a_composition_lemma_jacobian_identity.py`) establishes the single-hop
composition lemma as an identity `G(x_k) * J_k == G(x_{k+1})`, where `J_k` is
the reprojection Jacobian at hop `k` and `G` is the geometry term it rescales
(Bitterli/Veach form). T29 asks: does this survive being CHAINED across many
hops -- i.e. does `G(x_0) * Product_{k=0}^{T-1} J_k == G(x_T)` hold for
arbitrary hop count `T`, not just T=1?

**Proof by induction (mechanized below via SymPy, both an abstract-N `Product`
form and fully-expanded concrete-N sanity checks, mirroring
`test_a_merge_order_associativity.py`'s two-track style):**

  - Base case (T=1): `G(x_0)*J_0 == G(x_1)` -- exactly T27's own proven
    single-hop identity, taken here as the building block (not re-derived;
    re-deriving the real Bitterli/Veach `cos(theta)/dist^2` Jacobian algebra
    is T27's job, not this file's).
  - Inductive step: assume `G(x_0) * Product_{k=0}^{n-1} J_k == G(x_n)` holds.
    Multiply both sides by `J_n`:
    `G(x_0) * Product_{k=0}^{n} J_k == G(x_n) * J_n == G(x_{n+1})`
    (the last equality is T27's single-hop identity applied at hop `n`) --
    closing the induction for `n+1`.

Since each step only ever invokes the SAME single-hop identity applied at a
different index, the telescoping holds for ANY finite `T`, including T=500 --
this is a structural fact about the recursion (like T32's merge-order
invariance), not something that needs re-checking at every hop count the way
T29's own floating-point-robustness test does.
"""

import sympy as sp


def test_telescoping_holds_for_symbolic_hop_count():
    # Abstract-N form: G_k satisfies the single-hop identity G_k*J_k=G_{k+1}
    # for every k (T27's proven building block, taken as a hypothesis here),
    # expressed via IndexedBase so N itself stays symbolic. Working in log
    # space turns the telescoping PRODUCT claim into a telescoping SUM,
    # which SymPy's `Sum(...).doit()` resolves directly for an abstract
    # bound `n` (no per-N unrolling needed) -- the strongest form of this
    # proof, genuinely general in the hop count.
    n, k = sp.symbols("n k", integer=True, positive=True)
    G = sp.IndexedBase("G")

    # Hypothesis (T27): G[k]*J[k] = G[k+1] for every k, i.e.
    # ln(J[k]) = ln(G[k+1]) - ln(G[k]).
    log_J_from_hypothesis = sp.log(G[k + 1]) - sp.log(G[k])

    # Claim: Sum_{k=0}^{n-1} ln(J[k]) == ln(G[n]) - ln(G[0]), i.e.
    # G[0] * Product_{k=0}^{n-1} J[k] == G[n], for symbolic n.
    log_telescoped = sp.Sum(log_J_from_hypothesis, (k, 0, n - 1)).doit()
    log_target = sp.log(G[n]) - sp.log(G[0])
    assert sp.simplify(log_telescoped - log_target) == sp.Integer(0)


def test_telescoping_holds_for_concrete_hop_counts_fully_expanded():
    # Concrete-N re-derivation with fully expanded symbolic products (no
    # abstract Product/IndexedBase machinery) -- a second, independent code
    # path guarding against a false-positive from test 1's own machinery,
    # for hop counts spanning T29's actual tested range up to 500.
    for T in (1, 2, 3, 5, 10, 50, 200, 500):
        Gs = sp.symbols(f"G0:{T + 1}", positive=True)  # G_0 .. G_T

        # Impose the single-hop identity (T27) at every k: G_k*J_k = G_{k+1},
        # i.e. J_k = G_{k+1}/G_k. Check the telescoped product of these
        # per-hop Jacobians equals G_T exactly.
        j_from_hypothesis = [Gs[k + 1] / Gs[k] for k in range(T)]
        telescoped = Gs[0] * sp.prod(j_from_hypothesis)
        assert sp.simplify(telescoped - Gs[T]) == sp.Integer(0)


def test_chained_ratio_form_matches_direct_ratio_form_symbolically():
    # Mirrors test_t29's own "chained vs direct" comparison, but as an exact
    # symbolic identity rather than a ~1e-15 floating-point check: the
    # sequential product of per-hop RATIOS J_k/J_{k-1} telescopes to the
    # direct ratio J_T/J_0, for arbitrary symbolic per-hop Jacobians.
    for T in (1, 2, 5, 20, 500):
        Js = sp.symbols(f"J0:{T + 1}", positive=True)  # J_0 .. J_T
        chained = sp.prod(Js[k] / Js[k - 1] for k in range(1, T + 1))
        direct = Js[T] / Js[0]
        assert sp.simplify(chained - direct) == sp.Integer(0)


if __name__ == "__main__":
    test_telescoping_holds_for_symbolic_hop_count()
    test_telescoping_holds_for_concrete_hop_counts_fully_expanded()
    test_chained_ratio_form_matches_direct_ratio_form_symbolically()
    print("all A-composition-lemma-multihop-telescoping tests passed")
