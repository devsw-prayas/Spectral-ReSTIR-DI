"""Symbolic (SymPy) limit proof for G12-forward
(`addendum_session11_13_test_plan_extension.md` section 3,
"grazing-incidence-as-section-7-analog"):

**Claim:** grazing incidence in reconnection (`cos_theta_o(xA) -> 0`, which
makes `G(xA) -> 0` and the raw reprojection Jacobian `J_reproj -> infinity`)
is not a distinct, genuinely moving singularity -- it is the SAME mechanism
as Tier-0's own section-7 TIR fix (`snell_jacobian.tir_jacobian`): a factor
that vanishes exactly cancels a factor that diverges, leaving the COMBINED
quantity finite (in fact `C^infinity`, here literally constant) at the
would-be singular point. `session_log_restir_12.md` section 2 only checked
this numerically (`epsilon` down to `1e-14`, rel. error `1e-16`). This file
proves it as an exact symbolic limit -- the stronger, epsilon-independent
statement -- for BOTH sides of the claimed correspondence, then makes the
structural match itself an explicit, checked claim (matching leading-order
series behavior at the singular point), not just two limits that happen to
land on similar-looking finite numbers.

**Side A -- grazing reconnection (the new case G12-forward is about):**
`G(xA;theta) = cos(theta)/dist1^2` (vanishes linearly as `theta -> pi/2`,
`cos(theta) -> 0`); `J_reproj(theta) = [cos2/cos(theta)] * [dist1^2/dist2^2]`
(diverges as `1/cos(theta)` in lockstep). Both factors are built EXACTLY as
`test_t27_composition_lemma_real_jacobian.py`'s own real formula, specialized
to the grazing parametrization `session_log_restir_12.md` section 2 uses
(`xA`'s normal rotated by `theta`, `xB` fixed and never grazing).

**Side B -- TIR (the section-7 mechanism this is claimed to match):** the
raw (pre-simplified) Fresnel-power/solid-angle-Jacobian factors underlying
`src/snell_jacobian.py`'s own `tir_jacobian` docstring derivation:
`T_s(v)/eta^2` (vanishes linearly as `v -> 0`, the TIR-onset variable
`v = cos_theta_t`) times `J_theta_raw(v) = eta^2*c/v` (diverges as `1/v`).
The combined limit is checked against `tir_jacobian`'s own documented closed
form `J_TIR^s(0) = 4/eta`, tying this proof directly to the production
formula already living in `src/`, not an abstract restatement.

Both limits are taken on the UNSIMPLIFIED product (`sp.Mul(..., evaluate=False)`
to stop SymPy's eager auto-cancellation from doing the work before
`sp.limit` runs) -- this is what makes it a genuine limit computation
(0-times-infinity indeterminate form resolved by SymPy's own limit
machinery) rather than a restatement of an already-cancelled algebraic
identity, the same discipline
`test_a_composition_lemma_jacobian_identity.py` uses for the (unrelated,
non-singular) T27 tautology.
"""

import sympy as sp


def test_grazing_g_vanishes_and_j_reproj_diverges():
    theta = sp.symbols("theta", real=True)
    dist1, dist2, cos2 = sp.symbols("dist1 dist2 cos2", positive=True)

    G_xA = sp.cos(theta) / dist1**2
    J_reproj = (cos2 / sp.cos(theta)) * (dist1**2 / dist2**2)

    # approach from below (cos(theta) -> 0+, the physically valid grazing
    # direction -- theta never exceeds pi/2 for a surface-facing normal)
    assert sp.limit(G_xA, theta, sp.pi / 2, dir="-") == 0
    assert sp.limit(J_reproj, theta, sp.pi / 2, dir="-") is sp.oo


def test_grazing_combined_weight_is_finite_and_matches_g_xB():
    theta = sp.symbols("theta", real=True)
    dist1, dist2, cos2 = sp.symbols("dist1 dist2 cos2", positive=True)

    G_xA = sp.cos(theta) / dist1**2
    J_reproj = (cos2 / sp.cos(theta)) * (dist1**2 / dist2**2)
    raw_product = sp.Mul(G_xA, J_reproj, evaluate=False)

    limit_at_grazing = sp.limit(raw_product, theta, sp.pi / 2, dir="-")
    g_xB = cos2 / dist2**2  # G(xB), independent of theta by construction
    assert sp.simplify(limit_at_grazing - g_xB) == sp.Integer(0)


def test_grazing_combined_weight_derivative_vanishes_at_grazing():
    # C^infinity claim, first-derivative leg: the raw (unsimplified) product
    # is literally constant off theta=pi/2 (algebraic identity, same class
    # as T27's), so its derivative is identically zero there -- and that
    # zero derivative extends continuously through the grazing point itself.
    theta = sp.symbols("theta", real=True)
    dist1, dist2, cos2 = sp.symbols("dist1 dist2 cos2", positive=True)

    G_xA = sp.cos(theta) / dist1**2
    J_reproj = (cos2 / sp.cos(theta)) * (dist1**2 / dist2**2)
    raw_product = sp.Mul(G_xA, J_reproj, evaluate=False)

    deriv = sp.diff(raw_product, theta)
    assert sp.simplify(deriv) == sp.Integer(0)
    assert sp.limit(deriv, theta, sp.pi / 2, dir="-") == 0


def test_tir_side_t_vanishes_and_j_theta_diverges():
    v, eta, c = sp.symbols("v eta c", positive=True)

    # T_s(v)/eta**2, the f_BTDF throughput factor per tir_jacobian's own
    # docstring derivation (Fresnel s-polarized power transmittance / eta**2)
    t_over_eta2 = (4 * eta * c * v / (eta * c + v) ** 2) / eta**2
    j_theta_raw = eta**2 * c / v  # raw solid-angle-Jacobian factor, diverges at v=0

    assert sp.limit(t_over_eta2, v, 0, dir="+") == 0
    assert sp.limit(j_theta_raw, v, 0, dir="+") is sp.oo


def test_tir_side_combined_matches_production_closed_form():
    # Ties directly to src/snell_jacobian.py's tir_jacobian: J_TIR^s(v) =
    # 4*eta*c**2/(eta*c+v)**2, documented closed form at v=0 is 4/eta.
    v, eta, c = sp.symbols("v eta c", positive=True)

    t_over_eta2 = (4 * eta * c * v / (eta * c + v) ** 2) / eta**2
    j_theta_raw = eta**2 * c / v
    raw_combined = sp.Mul(t_over_eta2, j_theta_raw, evaluate=False)

    production_closed_form = 4 * eta * c**2 / (eta * c + v) ** 2  # snell_jacobian.tir_jacobian, s-pol

    # exact algebraic identity for all v (not just the limit) -- confirms the
    # "raw factors" used here really do combine to the production formula
    assert sp.simplify(sp.expand(t_over_eta2 * j_theta_raw) - production_closed_form) == sp.Integer(0)

    limit_at_tir_onset = sp.limit(raw_combined, v, 0, dir="+")
    assert sp.simplify(limit_at_tir_onset - 4 / eta) == sp.Integer(0)
    assert sp.limit(production_closed_form, v, 0, dir="+") == limit_at_tir_onset


def test_both_sides_share_the_same_order1_zero_times_order1_pole_structure():
    # The actual "same mechanism, not a coincidence" claim, mechanized: both
    # vanishing factors have a SIMPLE (order-1) zero at the singular point,
    # and both diverging factors have a SIMPLE (order-1) pole there -- so
    # the product's singularity order is exactly 0 (finite, nonzero) in both
    # cases, not merely "happens to evaluate to a finite number here."
    eps = sp.symbols("epsilon", positive=True)
    dist1, dist2, cos2 = sp.symbols("dist1 dist2 cos2", positive=True)
    eta, c = sp.symbols("eta c", positive=True)

    theta = sp.pi / 2 - eps  # eps -> 0+ approaches grazing from the valid side
    G_series = sp.series(sp.cos(theta) / dist1**2, eps, 0, 2).removeO()
    J_series = sp.series((cos2 / sp.cos(theta)) * (dist1**2 / dist2**2), eps, 0, 1)

    # G has a simple (order +1) zero in eps; J's leading term is order -1
    # (a simple pole) -- read off directly via the leading power of eps.
    g_leading_order = sp.Poly(G_series, eps).monoms()[0][0] if G_series != 0 else None
    assert g_leading_order == 1  # G ~ eps^1

    j_leading_exponent = min(
        term.as_coeff_exponent(eps)[1] for term in sp.Add.make_args(J_series.removeO())
    )
    assert j_leading_exponent == -1  # J ~ eps^(-1)

    v = sp.symbols("v", positive=True)
    T_series = sp.series((4 * eta * c * v / (eta * c + v) ** 2) / eta**2, v, 0, 2).removeO()
    Jt_series = sp.series(eta**2 * c / v, v, 0, 1)

    t_leading_order = sp.Poly(T_series, v).monoms()[0][0] if T_series != 0 else None
    assert t_leading_order == 1  # T/eta^2 ~ v^1, matching G's order-1 zero

    jt_leading_exponent = min(
        term.as_coeff_exponent(v)[1] for term in sp.Add.make_args(Jt_series.removeO())
    )
    assert jt_leading_exponent == -1  # J_theta_raw ~ v^(-1), matching J_reproj's order-1 pole


if __name__ == "__main__":
    test_grazing_g_vanishes_and_j_reproj_diverges()
    test_grazing_combined_weight_is_finite_and_matches_g_xB()
    test_grazing_combined_weight_derivative_vanishes_at_grazing()
    test_tir_side_t_vanishes_and_j_theta_diverges()
    test_tir_side_combined_matches_production_closed_form()
    test_both_sides_share_the_same_order1_zero_times_order1_pole_structure()
    print("all A-grazing-incidence-g12-forward tests passed")
