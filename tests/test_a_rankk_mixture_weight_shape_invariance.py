"""Symbolic (SymPy) proof of A13-forward's shape-invariance criterion:
rank-k reconnection triviality
(`T==id`, `J==1`) holds iff every species amplitude `a_i(lambda')` is
proportional to one shared shape -- the mixture-weight analog of A1's own
"lambda-invariance of `p(y|lambda)`" criterion, applied here to
`w_i(lambda') = a_i(lambda')*E_i / Sum_j a_j(lambda')*E_j` instead of a
single species' conditional shape. This claim was stated in prose
("Reconnection triviality holds iff every a_i is proportional to one shared
shape") but never mechanized -- this file supplies both directions:

**Forward direction (test 1-2).** If `a_i(lambda') = k_i*A(lambda')` for
every species `i` (a single shared shape `A`, differing only by per-species
constants `k_i`), then `A(lambda')` cancels between numerator and
denominator and `w_i(lambda') = k_i*E_i / Sum_j k_j*E_j` -- provably
CONSTANT, independent of `lambda'` -- mechanized both via `sympy.diff` (the
derivative w.r.t. `lambda'` is identically zero) and via direct algebraic
cancellation, for arbitrary species count `k`.

**Partition of unity (test 3).** `Sum_i w_i(lambda') == 1` always holds, for
the GENERAL (not necessarily shape-invariant) mixture-weight formula -- the
property A6's trichotomy proof needs the rank-k conditional to actually be a
valid probability mixture at all, mechanized here explicitly rather than
assumed.

**Negative direction, by counterexample (test 4).** When the `a_i` are NOT
all proportional to one shared shape, `w_i(lambda')` is not constant in
general -- confirmed via a concrete two-species counterexample with
genuinely different shapes (`a_1=lambda'`, `a_2=lambda'^2`, no common
factor), where `d(w_1)/d(lambda')` is symbolically nonzero. (This is a
counterexample, not a proof of the full converse for arbitrary non-
proportional families -- matching this repo's existing convention of using
one concrete counterexample to falsify a universal claim rather than
attempting a general non-constancy theorem, e.g.
`test_a_freepath_eventtype_score_identity.py`'s own negative control.)
"""

import sympy as sp


def test_shared_shape_makes_mixture_weights_constant_symbolic_general_n():
    n, i, j = sp.symbols("n i j", integer=True, positive=True)
    k = sp.IndexedBase("k")
    E = sp.IndexedBase("E")
    lam = sp.Symbol("lambda_prime", positive=True)
    A = sp.Function("A")(lam)  # shared shape, arbitrary functional form

    generic_sum = sp.Sum(k[j] * A * E[j], (j, 1, n))
    w_i = k[i] * A * E[i] / generic_sum

    # d(w_i)/d(lambda') == 0: the shared A(lambda') factor cancels between
    # numerator and every denominator term, so the ratio has no residual
    # lambda'-dependence at all.
    dw_i = sp.diff(w_i, lam)
    assert sp.simplify(dw_i) == sp.Integer(0)


def test_shared_shape_mixture_weight_equals_pure_confidence_share_concrete_n():
    for N in (1, 2, 3, 5, 8):
        ks = sp.symbols(f"k0:{N}", positive=True)
        Es = sp.symbols(f"E0:{N}", positive=True)
        A = sp.Symbol("A_val", positive=True)  # shared shape's value at some lambda'

        denom = sum(kk * A * Ee for kk, Ee in zip(ks, Es))
        for idx in range(N):
            w_i = ks[idx] * A * Es[idx] / denom
            confidence_share = ks[idx] * Es[idx] / sum(kk * Ee for kk, Ee in zip(ks, Es))
            assert sp.simplify(w_i - confidence_share) == sp.Integer(0)


def test_mixture_weights_sum_to_one_in_general_not_just_shape_invariant_case():
    # Sum_i w_i(lambda') == 1 for the GENERAL formula (distinct a_i(lambda'),
    # not assuming shared shape) -- required for A6's trichotomy to see a
    # valid probability mixture at all.
    n, i, j = sp.symbols("n i j", integer=True, positive=True)
    a = sp.IndexedBase("a")  # a[j] := a_j(lambda'), left fully general
    E = sp.IndexedBase("E")

    generic_sum = sp.Sum(a[j] * E[j], (j, 1, n))
    w = lambda idx: a[idx] * E[idx] / generic_sum

    for N in (1, 2, 3, 5):
        a_vals = sp.symbols(f"a0:{N}", positive=True)
        E_vals = sp.symbols(f"E0:{N}", positive=True)
        denom = sum(av * Ev for av, Ev in zip(a_vals, E_vals))
        total = sum(av * Ev / denom for av, Ev in zip(a_vals, E_vals))
        assert sp.simplify(total - 1) == sp.Integer(0)


def test_nonproportional_species_amplitudes_give_lambda_dependent_weight_counterexample():
    # Concrete counterexample: two species with NO common shape factor
    # (a_1=lambda', a_2=lambda'^2) -- w_1(lambda') is genuinely
    # lambda'-dependent, confirmed both symbolically (nonzero derivative as
    # a function) and at concrete points.
    lam = sp.Symbol("lambda_prime", positive=True)
    E1, E2 = sp.symbols("E1 E2", positive=True)
    a1, a2 = lam, lam ** 2

    w1 = a1 * E1 / (a1 * E1 + a2 * E2)
    dw1 = sp.simplify(sp.diff(w1, lam))
    assert dw1 != 0

    concrete = {E1: 2, E2: 3}
    w1_at_1 = sp.simplify(w1.subs(concrete).subs(lam, 1))
    w1_at_2 = sp.simplify(w1.subs(concrete).subs(lam, 2))
    assert w1_at_1 != w1_at_2  # genuinely moved as lambda' changed


if __name__ == "__main__":
    test_shared_shape_makes_mixture_weights_constant_symbolic_general_n()
    test_shared_shape_mixture_weight_equals_pure_confidence_share_concrete_n()
    test_mixture_weights_sum_to_one_in_general_not_just_shape_invariant_case()
    test_nonproportional_species_amplitudes_give_lambda_dependent_weight_counterexample()
    print("all A-rankk-mixture-weight-shape-invariance tests passed")
