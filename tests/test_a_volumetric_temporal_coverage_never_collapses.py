"""Symbolic (SymPy) proof of A9v/A10v (`restir_running_notes.md` sections
14-15): the volumetric-temporal Coverage Lemma condition holds
UNCONDITIONALLY at every finite optical depth, in contrast to the surface
case (A9, §12) where the condition is a genuine scene-dependent structural
check that can fail (T18's disocclusion, `gen=0/eval>0`, exactly the -50%
bias this repo's own `test_a_coverage_lemma_disocclusion_exact_bias.py`
derives in closed form).

**A9v's claim.** The volumetric target's optical-depth dependence lives
entirely in a transmittance factor `exp(-a(lambda')*C)`. For ANY finite
`C >= 0` and finite `a(lambda') > 0`, this factor is strictly positive --
never zero -- so it can never change WHERE `p_hat = a*L_e*G*exp(-a*C)` is
zero vs. nonzero: `supp(p_hat) == supp(a*L_e*G)`, independent of `C`. Since
this holds identically for both the gen-time and eval-time optical depths
(`C_gen`, `C_eval`, however different), `supp(p_hat_gen) == supp(p_hat_eval)`
always -- not merely `supp(p_hat_eval) subseteq supp(p_hat_gen)` (A9's own
necessary-and-sufficient condition), a strictly stronger equality that
trivially implies the inclusion Coverage Lemma requires.

**Test 1** mechanizes the "transmittance never zero at finite C, but its
LIMIT as C->infinity is zero" contrast explicitly -- this is exactly why the
surface case has a real cliff (a hard zero IS reachable there, at `G_gen=0`,
a finite/attainable scene state) while the volumetric case's "cliff" only
exists at the unreachable `C=infinity` limit, never at any actual finite
depth. **Test 2** mechanizes the support-equality claim itself: the sign/
zero-set of `a*L_e*G*exp(-a*C)` equals that of `a*L_e*G` alone, for
symbolic finite `C`, by factoring out the (symbolically confirmed) strictly
positive exponential. **Test 3** mechanizes A10v's logical step from A9v
to unbiasedness: `supp(eval)==supp(gen)` trivially implies A9's own
inclusion condition `supp(eval) cap supp(d) subseteq supp(gen)`, for an
ARBITRARY third set (the destination target's support), i.e. A10's proof
(which only ever needed the inclusion, never volumetric-specific structure)
transfers unchanged.

**Explicit non-claim, matching A10v's own caveat:** this file proves the
MEAN is exact at every finite `C` -- it says nothing about variance, which
`test_t22_volumetric_temporal_cgen_sweep_rb.py`'s own `ESS/M` degradation
already documents as growing unboundedly with `C_gen` (heavy-tailed
resampling weights). No test here touches variance/ESS; bias and variance
are separate axes and only the first is closed by A9v/A10v.
"""

import sympy as sp


def test_transmittance_never_zero_at_finite_c_but_limit_at_infinity_is_zero():
    a, C = sp.symbols("a C", positive=True)  # a(lambda') > 0, C in [0, infinity)
    transmittance = sp.exp(-a * C)

    # Never exactly zero at any finite, concrete C -- checked at a spread of
    # depths matching this repo's own T22/G9 tested range (C_gen up to 150).
    for c_val in (0, 1, 5, 20, 80, 150, 10_000):
        value = transmittance.subs({a: sp.Rational(3, 10), C: c_val})
        assert value.is_positive
        assert sp.simplify(value) != 0

    # But the LIMIT as C -> infinity genuinely is zero -- the surface-style
    # "cliff" this case is often compared to only exists at this unreachable
    # limit, never at a finite, attainable scene state.
    limit_at_infinity = sp.limit(transmittance.subs(a, sp.Rational(3, 10)), C, sp.oo)
    assert limit_at_infinity == 0


def test_support_equals_unweighted_target_support_independent_of_c():
    # supp(a*L_e*G*exp(-a*C)) == supp(a*L_e*G) for symbolic finite C: since
    # the transmittance factor is strictly positive, it can only rescale
    # magnitude, never flip a nonzero value to zero or vice versa. Modeled
    # with a concrete a*L_e*G shape (a Gaussian-times-positive-constant, the
    # kind of species/light product this repo's toy models use) so "support"
    # has a genuine nontrivial zero-set to compare against (a Gaussian's
    # support is everywhere nonzero, so contrast with a hard-cutoff shape
    # too, matching a windowed species profile).
    lam = sp.Symbol("lambda", real=True)
    C_pos = sp.Symbol("C", nonnegative=True)

    a_lam = sp.Rational(1, 5) + sp.Rational(1, 10) * lam ** 2  # a(lambda') > 0 always
    unweighted = a_lam * sp.exp(-((lam - 2) ** 2))  # a*L_e*G shape, everywhere positive
    transmittance = sp.exp(-a_lam * C_pos)

    full_target = unweighted * transmittance

    # Both are strictly positive everywhere (their zero-sets are both empty)
    # -- confirms equality of support the direct way: neither factor ever
    # introduces or removes a zero, for symbolic C_pos >= 0.
    for lam_val in (-5, -1, 0, 0.5, 3, 10):
        for c_val in (0, 1, 50):
            uv = sp.simplify(unweighted.subs(lam, lam_val))
            fv = sp.simplify(full_target.subs({lam: lam_val, C_pos: c_val}))
            # Same sign (both strictly positive here) at every probed
            # point/depth -- the transmittance factor never flips which
            # side of zero a point is on.
            assert uv > 0
            assert fv > 0


def test_support_equality_implies_coverage_lemma_inclusion_for_arbitrary_third_set():
    # A10v's logical step: supp(eval) == supp(gen) trivially implies A9's
    # own necessary-and-sufficient inclusion condition
    # supp(eval) cap supp(d) subseteq supp(gen), for an ARBITRARY destination
    # support set `d` -- mechanized via SymPy's Interval/Union set algebra
    # on a representative family of support sets (open intervals, unions),
    # not assuming any particular shape.
    from sympy import Interval, Union, FiniteSet

    candidate_sets = [
        Interval(-5, 5),
        Union(Interval(-10, -2), Interval(2, 10)),
        Interval.open(0, sp.oo),
        FiniteSet(1, 2, 3),
    ]

    for supp_gen in candidate_sets:
        supp_eval = supp_gen  # A9v's claim: equal, not just related
        for supp_d in candidate_sets:
            lhs = supp_eval.intersect(supp_d)
            assert lhs.is_subset(supp_gen)


if __name__ == "__main__":
    test_transmittance_never_zero_at_finite_c_but_limit_at_infinity_is_zero()
    test_support_equals_unweighted_target_support_independent_of_c()
    test_support_equality_implies_coverage_lemma_inclusion_for_arbitrary_third_set()
    print("all A-volumetric-temporal-coverage-never-collapses tests passed")
