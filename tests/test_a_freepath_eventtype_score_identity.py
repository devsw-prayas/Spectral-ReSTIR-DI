"""Symbolic (SymPy) proof of A11 (`restir_running_notes.md` section 17,
"Free-path/event-type score-term identity"): the value-level ratio identity
`E[1(i)*f_i(theta)/p_i(theta)] = Sum_i f_i(theta)` was already algebraically
obvious (p_i cancels for any theta). The actual open question in the running
notes is at the DERIVATIVE level -- for a differentiable-rendering-style
gradient estimator (autodiff through a stochastically-selected branch `i`),
does correctness require an explicit REINFORCE/score-function correction
term `Sum_i (d/dtheta ln p_i(theta)) * h_i(theta)`, or does the standard
inverse-pdf ratio structure `h_i(theta) = f_i(theta)/p_i(theta)` make it
unnecessary?

**The general REINFORCE/score-function identity** (exact, no assumptions on
`p_i`, standard total-derivative-of-an-expectation decomposition):

    d/dtheta [Sum_i p_i(theta)*h_i(theta)]
        = Sum_i p_i(theta) * [dh_i/dtheta + (d/dtheta ln p_i(theta)) * h_i(theta)]

i.e. the "pathwise" term (differentiate `h_i` holding the sampling weight
fixed) plus the "score" term (differentiate the log-density, weight by the
integrand) together reproduce the true total derivative. This is the
textbook identity being probed, not this repo's own claim -- test 1 below
just confirms SymPy agrees with it for a generic `p_i(theta)`, `h_i(theta)`.

**The A11-specific content** is what happens once `h_i := f_i/p_i` (the
inverse-pdf ratio structure named in the running notes):

    pathwise term:  p_i * d/dtheta[f_i/p_i] = f_i' - f_i*p_i'/p_i
    score term:     p_i * (p_i'/p_i) * (f_i/p_i) = f_i*p_i'/p_i

These two terms' `f_i*p_i'/p_i` pieces are EXACT NEGATIVES of each other --
they cancel identically, for any theta-dependence `p_i(theta)` at all, no
special structure needed beyond `h_i=f_i/p_i`. Test 2 mechanizes this
cancellation and confirms the sum collapses to `Sum_i f_i'(theta)`, matching
`d/dtheta[Sum_i f_i(theta)]` exactly -- the true target's derivative, with
both terms present.

**Why "no REINFORCE term needed... provided built correctly" is true, and
what "correctly" means:** if the estimator is implemented as detached-pdf
autodiff (the standard practice for inverse-pdf ratio estimators, and this
repo's actual convention, per `restir_running_notes.md`'s note that the
current oracle has no stochastic per-vertex event-type sampling at all) --
i.e. `p_i` is held CONSTANT in the differentiation graph even though its
value came from a theta-dependent formula -- then the score term is
identically zero (`p_i'=0` in the detached graph) and the pathwise term
ALONE already equals `f_i'(theta)` exactly (test 3): the score term was
never needed because it vanishes by construction, not because it was
correctly computed and added. Test 4 is the negative control: if `p_i(theta)`
is NOT detached and the score term is DROPPED anyway (a real differentiable-
rendering bug class -- autodiff naively through a stochastic branch without
the REINFORCE correction), the residual is exactly `-f_i(theta)*p_i'(theta)/p_i(theta)`,
which is nonzero in general -- confirmed both symbolically (nonzero as a
function) and via a concrete counterexample.
"""

import sympy as sp


def test_general_reinforce_identity_holds_for_arbitrary_p_and_h():
    # Textbook total-derivative-of-an-expectation decomposition, no
    # assumption on p_i/h_i's structure -- sanity anchor for tests 2-4.
    theta = sp.symbols("theta")
    n = 3
    p = [sp.Function(f"p{i}")(theta) for i in range(n)]
    h = [sp.Function(f"h{i}")(theta) for i in range(n)]

    lhs = sp.diff(sum(p[i] * h[i] for i in range(n)), theta)
    rhs = sum(
        p[i] * (sp.diff(h[i], theta) + (sp.diff(p[i], theta) / p[i]) * h[i])
        for i in range(n)
    )
    assert sp.simplify(lhs - rhs) == sp.Integer(0)


def test_pathwise_and_score_terms_cancel_for_inverse_pdf_ratio_structure():
    # A11's core cancellation: h_i := f_i/p_i makes the score term's
    # f_i*p_i'/p_i piece an exact negative of the pathwise term's own
    # f_i*p_i'/p_i piece, for a GENERAL symbolic theta-dependent p_i(theta).
    theta = sp.symbols("theta")
    n = 4
    p = [sp.Function(f"p{i}")(theta) for i in range(n)]
    f = [sp.Function(f"f{i}")(theta) for i in range(n)]
    h = [f[i] / p[i] for i in range(n)]

    pathwise_term = sum(p[i] * sp.diff(h[i], theta) for i in range(n))
    score_term = sum(p[i] * (sp.diff(p[i], theta) / p[i]) * h[i] for i in range(n))
    total = sp.simplify(pathwise_term + score_term)

    truth_derivative = sp.simplify(sp.diff(sum(f), theta))
    assert sp.simplify(total - truth_derivative) == sp.Integer(0)


def test_detached_pdf_pathwise_term_alone_already_correct():
    # "Built correctly" = p_i held CONSTANT in the diff graph (detached),
    # even though it numerically came from a theta-dependent formula. Then
    # the score term is identically zero (p_i'=0) and the pathwise term
    # alone -- no REINFORCE correction added -- already equals the true
    # derivative exactly. This is the actual claim in restir_running_notes.md
    # section 17 ("no REINFORCE term needed if built correctly").
    theta = sp.symbols("theta")
    n = 5
    p_const = sp.symbols(f"p0:{n}", positive=True)  # detached: plain constants
    f = [sp.Function(f"f{i}")(theta) for i in range(n)]

    pathwise_only = sum(p_const[i] * sp.diff(f[i] / p_const[i], theta) for i in range(n))
    truth_derivative = sp.diff(sum(f), theta)
    assert sp.simplify(pathwise_only - truth_derivative) == sp.Integer(0)


def test_nondetached_pdf_dropping_score_term_is_generally_wrong():
    # Negative control: if p_i(theta) genuinely depends on theta (not
    # detached) and the score term is dropped anyway, the pathwise-only
    # sum is short by exactly Sum_i f_i(theta)*p_i'(theta)/p_i(theta) --
    # confirmed symbolically nonzero as a function, then with a concrete
    # counterexample at numeric theta.
    theta = sp.symbols("theta")
    n = 2
    p = [sp.Function(f"p{i}")(theta) for i in range(n)]
    f = [sp.Function(f"f{i}")(theta) for i in range(n)]
    h = [f[i] / p[i] for i in range(n)]

    pathwise_only = sum(p[i] * sp.diff(h[i], theta) for i in range(n))
    truth_derivative = sp.diff(sum(f), theta)
    residual = sp.simplify(pathwise_only - truth_derivative)

    expected_residual = sp.simplify(
        -sum(f[i] * sp.diff(p[i], theta) / p[i] for i in range(n))
    )
    assert sp.simplify(residual - expected_residual) == sp.Integer(0)

    # Concrete counterexample: p_0(theta)=theta/(theta+1), p_1=1/(theta+1)
    # (a genuine theta-dependent two-outcome pdf), f_0=theta^2, f_1=3*theta.
    concrete = {
        p[0]: theta / (theta + 1),
        p[1]: 1 / (theta + 1),
        f[0]: theta ** 2,
        f[1]: 3 * theta,
    }
    residual_at_2 = sp.simplify(expected_residual.subs(concrete)).subs(theta, 2)
    assert residual_at_2 != 0


if __name__ == "__main__":
    test_general_reinforce_identity_holds_for_arbitrary_p_and_h()
    test_pathwise_and_score_terms_cancel_for_inverse_pdf_ratio_structure()
    test_detached_pdf_pathwise_term_alone_already_correct()
    test_nondetached_pdf_dropping_score_term_is_generally_wrong()
    print("all A-freepath-eventtype-score-identity tests passed")
