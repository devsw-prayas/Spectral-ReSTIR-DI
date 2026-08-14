"""Symbolic (SymPy) proof extending the composition lemma's multi-hop
telescoping (`test_a_composition_lemma_multihop_telescoping.py`, backing
T29) to T28's actual setting: a genuine 2D area Jacobian, not a 1D scalar
ratio. `test_t28_composition_lemma_2d_deforming_surface.py` computes its
area Jacobians via CENTRAL FINITE DIFFERENCES on two concrete embeddings
(explicitly "not hand-derived, to avoid calculus errors") and only ever
checks a single A->B hop -- it never establishes, symbolically or otherwise,
that chaining several 2D reparametrizations composes multiplicatively the
way T29 proved for the 1D case.

**The claim.** For 2D reparametrization maps `phi_1: domain_0 -> domain_1`
and `phi_2: domain_1 -> domain_2` (both genuine `R^2 -> R^2` maps with real
shear, matching T28's own `u + c*sin(pi*v)`-style embedding), the composed
map's Jacobian DETERMINANT satisfies the multivariate chain rule:

    det(D(phi_2 . phi_1))(u,v) == det(D phi_2)(phi_1(u,v)) * det(D phi_1)(u,v)

i.e. area-Jacobian ratios compose by multiplication under composition, the
exact 2D analog of T29's 1D telescoping product -- and, chained over K hops,
generalizes to `det(D(phi_K . ... . phi_1)) == Product_k det(D phi_k)`, the
genuine multi-hop version of T28's own single-hop `J_area,B/J_area,A` ratio.

**Mechanized via SymPy's exact symbolic Jacobian/determinant machinery**
(`Matrix.jacobian`, `Matrix.det`), not finite differences -- test 3 then
cross-checks the exact symbolic result against T28's own central-finite-
difference methodology at concrete points, confirming FD is a good
approximation of the true (now exactly known) area-Jacobian ratio, not an
independent source of truth in its own right.
"""

import sympy as sp


def _shear_map(u, v, amp_u, amp_v, freq=sp.pi):
    """A genuine 2D shear reparametrization, same functional family as
    T28's XB embedding's in-plane shear (`u + c*sin(pi*v)`,
    `v + c*sin(pi*u)`)."""
    return sp.Matrix([u + amp_u * sp.sin(freq * v), v + amp_v * sp.sin(freq * u)])


def test_two_hop_jacobian_determinant_chain_rule():
    u, v = sp.symbols("u v", real=True)

    phi1 = _shear_map(u, v, sp.Rational(1, 4), sp.Rational(3, 20))
    J_phi1 = phi1.jacobian([u, v])
    det_phi1 = sp.simplify(J_phi1.det())

    p, q = sp.symbols("p q", real=True)
    phi2 = _shear_map(p, q, sp.Rational(3, 10), sp.Rational(1, 8))
    J_phi2 = phi2.jacobian([p, q])
    det_phi2_at_pq = sp.simplify(J_phi2.det())
    det_phi2_at_phi1 = det_phi2_at_pq.subs({p: phi1[0], q: phi1[1]})

    composed = phi2.subs({p: phi1[0], q: phi1[1]})
    J_composed = composed.jacobian([u, v])
    det_composed = sp.simplify(J_composed.det())

    chain_rule_rhs = sp.simplify(det_phi2_at_phi1 * det_phi1)

    assert sp.simplify(det_composed - chain_rule_rhs) == sp.Integer(0)


def test_three_hop_jacobian_determinant_telescopes():
    # phi_3 . phi_2 . phi_1: det(D(phi_3.phi_2.phi_1)) ==
    # det(Dphi_3)(phi_2(phi_1)) * det(Dphi_2)(phi_1) * det(Dphi_1) --
    # genuine 3-hop 2D reparametrization chain, three distinct shear maps.
    u, v = sp.symbols("u v", real=True)
    p, q = sp.symbols("p q", real=True)
    r, s = sp.symbols("r s", real=True)

    phi1 = _shear_map(u, v, sp.Rational(1, 5), sp.Rational(1, 6))
    phi2 = _shear_map(p, q, sp.Rational(1, 4), sp.Rational(3, 20))
    phi3 = _shear_map(r, s, sp.Rational(3, 10), sp.Rational(1, 8))

    det1_uv = sp.simplify(phi1.jacobian([u, v]).det())
    det2_pq = sp.simplify(phi2.jacobian([p, q]).det())
    det3_rs = sp.simplify(phi3.jacobian([r, s]).det())

    stage1 = phi1  # domain_0 -> domain_1, in terms of (u,v)
    det2_at_stage1 = det2_pq.subs({p: stage1[0], q: stage1[1]})

    stage2 = phi2.subs({p: stage1[0], q: stage1[1]})  # domain_0 -> domain_2
    det3_at_stage2 = det3_rs.subs({r: stage2[0], s: stage2[1]})

    composed = phi3.subs({r: stage2[0], s: stage2[1]})  # domain_0 -> domain_3
    det_composed = sp.simplify(composed.jacobian([u, v]).det())

    chain_rule_rhs = sp.simplify(det3_at_stage2 * det2_at_stage1 * det1_uv)

    assert sp.simplify(det_composed - chain_rule_rhs) == sp.Integer(0)


def test_exact_symbolic_jacobian_matches_t28_style_finite_difference():
    # Cross-check: T28's own methodology (central finite differences on a
    # concrete embedding, "not hand-derived, to avoid calculus errors") must
    # agree with the now-exactly-known symbolic Jacobian determinant at
    # concrete points -- confirms FD is validly approximating a real smooth
    # quantity, not silently masking a different one.
    u, v = sp.symbols("u v", real=True)
    phi = _shear_map(u, v, sp.Rational(1, 4), sp.Rational(3, 20))
    exact_det = sp.lambdify((u, v), sp.simplify(phi.jacobian([u, v]).det()), "sympy")

    phi_num = sp.lambdify((u, v), phi, "sympy")

    def fd_det(u0, v0, h=sp.Rational(1, 100_000)):
        du = (sp.Matrix(phi_num(u0 + h, v0)) - sp.Matrix(phi_num(u0 - h, v0))) / (2 * h)
        dv = (sp.Matrix(phi_num(u0, v0 + h)) - sp.Matrix(phi_num(u0, v0 - h))) / (2 * h)
        return sp.Matrix.hstack(du, dv).det()

    for u0, v0 in [(sp.Rational(1, 5), sp.Rational(3, 10)), (sp.Rational(4, 5), sp.Rational(1, 5)),
                   (sp.Rational(1, 2), sp.Rational(1, 2))]:
        exact_val = sp.N(exact_det(u0, v0))
        fd_val = sp.N(fd_det(u0, v0))
        assert abs(exact_val - fd_val) < sp.Float(1e-6)


if __name__ == "__main__":
    test_two_hop_jacobian_determinant_chain_rule()
    test_three_hop_jacobian_determinant_telescopes()
    test_exact_symbolic_jacobian_matches_t28_style_finite_difference()
    print("all A-composition-lemma-2d-jacobian-chain-rule tests passed")
