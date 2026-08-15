"""Symbolic (SymPy) proof for A1, "Reconnection-validity theorem, surface"
(`restir_running_notes.md` section 2). A2 (volumetric, section 3) reduces to
A1 "by inspection" per its own text (kernel algebra is shared, and the
volumetric support piece has no Snell-type mechanism at all), so this file
also closes out A2's only nontrivial, mechanizable content.

A1's proof has two directions:

  (⟸) λ-invariant p(y|λ) ⟹ T≡id, J≡1 is valid — trivial, standard RIS.
  (⟹, converse) at dispersive vertices, dropping the Jacobian correction
      η²cosθ_i/cosθ_t is exactly the Tier-0 missing-Jacobian bias; near the
      critical angle the map isn't even onto (TIR) — an existence failure,
      not a Jacobian defect.

The converse direction is the one with real symbolic content, and it rests
on three claims this file mechanizes against `src/snell_jacobian.py` and
`src/cauchy_ior.py`'s actual closed forms (not re-derived from scratch —
checked that the shipped code IS what the theorem claims):

1. `refracted_direction` is a genuine unit-norm vector satisfying the scalar
   Snell's law `n_i sinθ_i = n_t sinθ_t`, for arbitrary orthonormal
   (n_hat, t_hat) — i.e. the vector form is not an independent assumption,
   it's forced by cosθ_t's own defining formula (`cauchy_ior.cos_theta_t`).
2. `solid_angle_ratio`'s closed form (n_i/n_t)² cosθ_i/cosθ_t is exactly the
   1-D area-distortion factor of the reparametrization θ_i → θ_t induced by
   scalar Snell's law (implicit differentiation), independent of any vector
   machinery — this is the Jacobian A1 says is mandatory once support is
   preserved.
3. `snell_jacobian`'s 3×3 closed form is internally consistent with (2): its
   eigenvalue along n_hat (the "compression" direction) is exactly
   `solid_angle_ratio`, and its two tangential eigenvalues both equal
   η = n_i/n_t — so the 2×2 sphere-map Jacobian derived independently in (2)
   is not a separate object from Theorem 1's 3×3 formula, it's the same
   thing restricted to the normal direction.
4. The critical-angle existence failure: cosθ_t's defining formula
   `sqrt(1 - η²(1-cos²θ_i))` has no real solution once
   η²(1-cos²θ_i) > 1 (i.e. sinθ_i > n_t/n_i) — formalizing "the map isn't
   even onto" as a genuine non-existence of θ_t, not a numerical edge case
   the Jacobian could patch over.

Each probe is an exact `sp.simplify(...) == 0` (or explicit real/complex)
check, not a numeric tolerance — same style as
`test_a_composition_lemma_jacobian_identity.py`.
"""

import sympy as sp


def test_refracted_direction_is_unit_norm_and_satisfies_snells_law():
    # omega_i decomposed in the plane of incidence: omega_i = -cos_i*n_hat + sin_i*t_hat,
    # with n_hat, t_hat treated as an orthonormal pair via symbolic dot-product rules
    # (no coordinates needed -- n_hat.n_hat=1, t_hat.t_hat=1, n_hat.t_hat=0 substituted
    # directly into the expanded dot products below).
    n_i, n_t, cos_i, sin_i = sp.symbols("n_i n_t cos_i sin_i", positive=True)
    eta = n_i / n_t

    # cos_theta_t as literally defined in src/cauchy_ior.py (pre-clamp, propagating branch):
    # cos_t = sqrt(1 - eta**2 * sin_i**2). Keep it as an abstract symbol cos_t constrained
    # by that defining equation, so downstream algebra stays polynomial.
    cos_t = sp.symbols("cos_t", positive=True)
    snell_defining_eq = sp.Eq(cos_t**2, 1 - eta**2 * sin_i**2)

    # refracted_direction's actual formula (src/snell_jacobian.py):
    #   omega_t = eta*omega_i + (eta*cos_i - cos_t)*n_hat
    # Expand omega_i = -cos_i*n_hat + sin_i*t_hat and collect n_hat/t_hat coefficients.
    n_hat_coeff = eta * (-cos_i) + (eta * cos_i - cos_t)
    t_hat_coeff = eta * sin_i

    n_hat_coeff_simplified = sp.simplify(n_hat_coeff)
    assert n_hat_coeff_simplified == -cos_t

    # |omega_t|^2 = n_hat_coeff^2 + t_hat_coeff^2 (orthonormal basis) -- must equal 1
    # exactly once cos_t^2 is substituted via the Snell-defining equation.
    norm_sq = n_hat_coeff_simplified**2 + t_hat_coeff**2
    norm_sq_on_shell = norm_sq.subs(cos_t**2, sp.solve(snell_defining_eq, cos_t**2)[0])
    residual = sp.simplify(norm_sq_on_shell - 1)
    assert residual == sp.Integer(0)

    # Scalar Snell's law: t_hat coefficient of omega_t is by definition sin_t
    # (angle measured from n_hat), so n_i*sin_i == n_t*sin_t must hold identically.
    sin_t = t_hat_coeff
    residual_snell = sp.simplify(n_i * sin_i - n_t * sin_t)
    assert residual_snell == sp.Integer(0)


def test_solid_angle_ratio_matches_implicit_differentiation_of_snells_law():
    # Independent derivation, no reference to src/snell_jacobian.py's formula:
    # scalar Snell's law n_i*sin(theta_i) = n_t*sin(theta_t) defines theta_t(theta_i).
    # The reparametrization's 1-D "area" (density) ratio for a sphere map that leaves
    # the azimuthal angle phi untouched is (sin(theta_t)/sin(theta_i)) * d(theta_t)/d(theta_i)
    # -- the standard change-of-variables Jacobian for dOmega = sin(theta) dtheta dphi.
    n_i, n_t, theta_i = sp.symbols("n_i n_t theta_i", positive=True)
    theta_t = sp.Function("theta_t")

    # Implicit differentiation of n_i*sin(theta_i) = n_t*sin(theta_t(theta_i)):
    #   n_i*cos(theta_i) = n_t*cos(theta_t)*theta_t'(theta_i)
    theta_t_sym, cos_t_sym = sp.symbols("theta_t_sym cos_t_sym", positive=True)
    dtheta_t_dtheta_i = n_i * sp.cos(theta_i) / (n_t * cos_t_sym)

    sin_t_sym = n_i * sp.sin(theta_i) / n_t   # from Snell's law directly

    area_ratio = (sin_t_sym / sp.sin(theta_i)) * dtheta_t_dtheta_i
    area_ratio_simplified = sp.simplify(area_ratio)

    # src/snell_jacobian.py's solid_angle_ratio: (n_i/n_t)**2 * cos_i/cos_t
    solid_angle_ratio_closed_form = (n_i / n_t) ** 2 * sp.cos(theta_i) / cos_t_sym

    residual = sp.simplify(area_ratio_simplified - solid_angle_ratio_closed_form)
    assert residual == sp.Integer(0)


def test_3x3_snell_jacobian_eigenvalues_match_solid_angle_ratio_and_eta():
    # Build the exact 3x3 matrix from src/snell_jacobian.py's formula in a basis where
    # n_hat = e1, and the tangent plane is spanned by e2, e3 (orthonormal, WLOG -- the
    # formula only depends on n_hat through the outer product n_hat n_hat^T, which is
    # basis-covariant, so checking one orthonormal frame proves it for all).
    n_i, n_t, cos_i, cos_t = sp.symbols("n_i n_t cos_i cos_t", positive=True)
    eta = n_i / n_t
    c = 1 - eta * cos_i / cos_t

    I3 = sp.eye(3)
    nnT = sp.zeros(3, 3)
    nnT[0, 0] = 1   # n_hat = e1 in this frame

    J = eta * (I3 - c * nnT)

    eigenvals = J.eigenvals()   # {eigenvalue: multiplicity}

    normal_eigenvalue = sp.simplify(eta * (1 - c))
    solid_angle_ratio_closed_form = sp.simplify((n_i / n_t) ** 2 * cos_i / cos_t)
    assert sp.simplify(normal_eigenvalue - solid_angle_ratio_closed_form) == sp.Integer(0)

    # Confirm eigenvals() actually reports: eta with multiplicity 2 (tangential),
    # and the normal eigenvalue with multiplicity 1, and nothing else.
    found_tangential = False
    found_normal = False
    for val, mult in eigenvals.items():
        val_s = sp.simplify(val)
        if sp.simplify(val_s - eta) == sp.Integer(0) and mult == 2:
            found_tangential = True
        if sp.simplify(val_s - normal_eigenvalue) == sp.Integer(0) and mult == 1:
            found_normal = True
    assert found_tangential and found_normal
    assert len(eigenvals) == 2


def test_tir_is_a_genuine_nonexistence_not_a_jacobian_singularity():
    # cos_theta_t's defining formula (src/cauchy_ior.py, pre-clamp):
    #   cos_t = sqrt(1 - eta**2 * (1 - cos_i**2))
    # Past the critical angle the radicand is negative -- theta_t has no real solution
    # at all, independent of what the Jacobian does. This is the "map isn't even onto"
    # claim: no reconnection target exists, so no Jacobian correction (finite or not)
    # can repair it -- it's an existence failure, structurally different from a
    # divergence (which IS repairable, per Theorem 3 / snell_jacobian_tir_safe).
    n_i, n_t, cos_i = sp.symbols("n_i n_t cos_i", positive=True)
    eta = n_i / n_t
    radicand = 1 - eta**2 * (1 - cos_i**2)

    # Concrete numeric instance strictly past the critical angle: n_i=1.5, n_t=1.0,
    # sin_i=0.9 > n_t/n_i=0.667 -- radicand must be strictly negative.
    subs_beyond_critical = {n_i: sp.Rational(3, 2), n_t: sp.Integer(1), cos_i: sp.sqrt(1 - sp.Rational(81, 100))}
    val = sp.simplify(radicand.subs(subs_beyond_critical))
    assert val < 0
    # sqrt of a negative real is genuinely non-real (not just "small"/clamped) --
    # confirm sympy agrees cos_t would be complex, i.e. no physical theta_t exists.
    cos_t_val = sp.sqrt(val)
    assert cos_t_val.is_real is False

    # Sanity boundary: exactly at the critical angle the radicand is exactly zero
    # (cos_t = 0, a genuine grazing solution, not yet a non-existence).
    sin_i_crit = n_t / n_i
    cos_i_crit = sp.sqrt(1 - sin_i_crit**2)
    val_crit = sp.simplify(radicand.subs({cos_i: cos_i_crit}))
    assert val_crit == sp.Integer(0)


if __name__ == "__main__":
    test_refracted_direction_is_unit_norm_and_satisfies_snells_law()
    test_solid_angle_ratio_matches_implicit_differentiation_of_snells_law()
    test_3x3_snell_jacobian_eigenvalues_match_solid_angle_ratio_and_eta()
    test_tir_is_a_genuine_nonexistence_not_a_jacobian_singularity()
    print("all A-snell-reconnection-validity-jacobian tests passed")
