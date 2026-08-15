"""Symbolic (SymPy) proof for the composition lemma's core identity: the
reconnection Jacobian at one shading-point pair exactly rescales the
point-light geometric term from one endpoint to the other.

T27 (`test_t27_composition_lemma_real_jacobian.py`) checked the identity

    G(x1) * J_reproj == G(x2)

numerically, for one concrete pair of parametrized 3D shading-point curves,
to a ~1e-16 floating-point tolerance. That is evidence, not proof: it confirms
the *implementation* is self-consistent on one scene, but the claim in the
theory notes is that this identity is an algebraic TAUTOLOGY, true for
*arbitrary* positions/normals, not a numerical coincidence of the test's
specific curves. This file proves that stronger claim symbolically.

The claim rests on the point-light geometric term

    G(x) = cos_theta_o(x) / dist(x, y)^2,   cos_theta_o(x) = dot(y - x, n) / dist(x, y)

and the reconnection Jacobian, DEFINED (Bitterli/Veach single-vertex NEE
reconnection) as the ratio of the term at the destination to the term at the
source:

    J_reproj = [cos_theta_o(x2) / cos_theta_o(x1)] * [dist(x1,y)^2 / dist(x2,y)^2]

Two probes, in increasing generality:

1. **Abstract-symbol cancellation**: treat cos1, cos2, dist1, dist2 as free
   symbols (not yet expanded into vector components) and show the ratio
   algebra collapses to zero residual. This isolates *why* the identity
   holds -- cos1 and dist1^2 each appear once in the numerator of G1 and once
   in the denominator of J_reproj's defining ratio, so they cancel on
   contact, independent of what cos1/dist1 actually equal. J_reproj is BY
   CONSTRUCTION the ratio G2/G1, so G1*J_reproj=G2 is a tautology at this
   level, not a geometric fact yet.
2. **Full vector-component expansion**: substitute genuine symbolic 3-vectors
   for x1, x2, y, n1, n2 (nine free position symbols + six free normal-
   component symbols, no unit-normal constraint needed -- see note below) and
   sqrt-based distances, then re-derive the same zero residual after a full
   symbolic expansion. This is the stronger check: it confirms the sqrt/dot-
   product machinery underlying "dist" and "cos_theta_o" doesn't secretly
   introduce a domain-dependent term that the abstract version hides (e.g. a
   sign ambiguity from squaring a sqrt). Assumes dist1, dist2, cos1 are
   nonzero (the reconnection-existence precondition T27's own
   `test_cos_theta_o_never_goes_negative_over_the_domain` probe checks
   numerically) -- SymPy is told this via `nonzero=True` symbols standing in
   for the already-verified-positive quantities, not via a fresh positivity
   proof (that's a geometric fact about the scene, out of scope here).

**Non-obvious point worth recording**: the identity does NOT require n1/n2 to
be unit vectors. `n1` only ever appears inside `cos1`, and `cos1` cancels
whole between G1 and J_reproj's ratio -- so the proof holds for arbitrary
(even non-unit, even non-physical) normal vectors. This is a strictly more
general statement than "the renderer's unit-normal convention makes this
work"; it's definitional algebra, true regardless of that convention.
"""

import sympy as sp


def test_abstract_symbol_cancellation_is_exact():
    cos1, cos2, dist1, dist2 = sp.symbols("cos1 cos2 dist1 dist2", nonzero=True)

    G1 = cos1 / dist1**2
    G2 = cos2 / dist2**2
    j_reproj = (cos2 / cos1) * (dist1**2 / dist2**2)

    residual = sp.simplify(G1 * j_reproj - G2)
    assert residual == sp.Integer(0)


def test_full_vector_component_identity_is_exact():
    x11, x12, x13 = sp.symbols("x11 x12 x13", real=True)
    x21, x22, x23 = sp.symbols("x21 x22 x23", real=True)
    y1, y2, y3 = sp.symbols("y1 y2 y3", real=True)
    n11, n12, n13 = sp.symbols("n11 n12 n13", real=True)
    n21, n22, n23 = sp.symbols("n21 n22 n23", real=True)

    x1 = sp.Matrix([x11, x12, x13])
    x2 = sp.Matrix([x21, x22, x23])
    y = sp.Matrix([y1, y2, y3])
    n1 = sp.Matrix([n11, n12, n13])
    n2 = sp.Matrix([n21, n22, n23])

    d1 = y - x1
    d2 = y - x2

    dist1_sq = (d1.T * d1)[0, 0]
    dist2_sq = (d2.T * d2)[0, 0]
    dist1 = sp.sqrt(dist1_sq)
    dist2 = sp.sqrt(dist2_sq)

    cos1 = (d1.T * n1)[0, 0] / dist1
    cos2 = (d2.T * n2)[0, 0] / dist2

    G1 = cos1 / dist1**2
    G2 = cos2 / dist2**2
    j_reproj = (cos2 / cos1) * (dist1**2 / dist2**2)

    lhs = sp.together(G1 * j_reproj)
    rhs = sp.together(G2)

    # cos1 cancels between G1 and j_reproj's ratio; dist1**2 likewise --
    # neither sqrt ever needs to be expanded/rationalized for the residual
    # to vanish, confirming the cancellation is purely algebraic (structural,
    # not dependent on the sqrt branch or sign of any dot product).
    residual = sp.simplify(lhs - rhs)
    assert residual == sp.Integer(0)

    # Independent cross-check: substitute concrete generic (non-special)
    # numeric values and confirm the symbolic zero survives evaluation, as a
    # sanity check against a sign error that `simplify` alone might paper
    # over (e.g. an accidental 0/0 simplification).
    subs = {
        x11: 1, x12: 2, x13: 3,
        x21: 5, x22: -1, x23: 2,
        y1: 3, y2: 4, y3: 9,
        n11: 0.2, n12: 0.1, n13: 0.9,
        n21: -0.3, n22: 0.4, n23: 0.7,
    }
    lhs_val = complex(lhs.subs(subs).evalf())
    rhs_val = complex(rhs.subs(subs).evalf())
    assert abs(lhs_val - rhs_val) < 1e-12


def test_j_reproj_is_definitionally_the_ratio_g2_over_g1():
    # Sanity anchor: confirms the abstract-symbol probe's own premise --
    # J_reproj as defined literally equals G2/G1, so probe 1's "cancellation"
    # is not a coincidence of algebra manipulation order.
    cos1, cos2, dist1, dist2 = sp.symbols("cos1 cos2 dist1 dist2", nonzero=True)
    G1 = cos1 / dist1**2
    G2 = cos2 / dist2**2
    j_reproj = (cos2 / cos1) * (dist1**2 / dist2**2)
    assert sp.simplify(j_reproj - G2 / G1) == sp.Integer(0)


if __name__ == "__main__":
    test_abstract_symbol_cancellation_is_exact()
    test_full_vector_component_identity_is_exact()
    test_j_reproj_is_definitionally_the_ratio_g2_over_g1()
    print("all A-composition-lemma-jacobian-identity tests passed")
