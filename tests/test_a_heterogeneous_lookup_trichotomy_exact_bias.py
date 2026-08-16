"""Symbolic (SymPy) proof of A6 (heterogeneous local-lookup consistency
trichotomy, running_notes Sec. 7) and A7 (free-path firewall corollary,
Sec. 9), in exact closed form.

T7-T10 numerically confirmed the three-way trichotomy (naive / IS-reweight /
fix-local) and A6/A7's prose proof states the qualitative conditions
(naive biased whenever `q_zA != q_zB`; IS-reweight unbiased iff support
containment holds, else *structurally* biased; fix-local always unbiased,
independent of the source vertex's distribution). None of that was ever
reduced to an exact algebraic expression the way A9's Coverage Lemma
disocclusion bias was (`test_a_coverage_lemma_disocclusion_exact_bias.py`).
This file supplies that: a minimal discrete 2-species toy (no lambda'
integral, since the trichotomy's mechanism only depends on the *species*
weights, not the emission/absorption shapes) admits closed forms for all
three strategies' expectations, in exactly the same "derive symbolically,
then cross-check the real module + quadrature at a concrete config" style.

**Toy setup.** Species `j in {1,2}`, un-normalized target values `t1,t2>0`
at destination `z_B` (`p_hat(j;z_B)=t_j`), normalized proposals
`q_A(1)=a, q_A(2)=1-a` at the source and `q_B(1)=b, q_B(2)=1-b` at the
destination. `TRUTH_B = t1+t2`.

- **Naive** (`score=p_hat(y;z_B)/q_B(y)`, sampled from `q_A`):
  `E = a*t1/b + (1-a)*t2/(1-b)`, which equals `TRUTH_B` only at `a==b` (or
  the degenerate `t1==t2` case) -- a genuine closed-form bias whenever the
  two vertices' proposals differ.
- **IS-reweight** (`score=p_hat(y;z_B)/q_A(y)`, sampled from `q_A`):
  `E = t1+t2` identically, for ANY `a` strictly between 0 and 1 -- the
  standard importance-sampling identity, unconditionally exact regardless
  of the `a` vs `b` mismatch, as long as support holds.
- **IS-reweight, support violated** (`a=0`, species 2 dead at the source):
  only species 1 can ever be drawn, so `E = t1`, an exact `-t2/(t1+t2)`
  relative bias -- structural, matching T10's -78.4%-class finding, and the
  same "-x_dead/(x_dead+x_alive)" shape as A9's disocclusion formula.
- **Fix-local** (`score=p_hat(y;z_B)/q_B(y)`, sampled from `q_B` itself):
  `E = t1+t2` identically, and critically the expression never contains `a`
  at all -- this is A7's tower-property claim made literal: fix-local's
  unbiasedness proof uses no property whatsoever of the source vertex's
  distribution.

Final test cross-checks the support-violation bias formula against the real
`heterogeneous_lookup.is_reweight_score` + quadrature-computed per-species
truth, at T10's own hard-cutoff configuration.
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import sympy as sp
import torch

torch.set_default_dtype(torch.float64)

from heterogeneous_lookup import (
    species_weight,
    local_target,
    is_reweight_score,
)


def test_naive_exact_bias_formula_symbolic():
    a, b, t1, t2 = sp.symbols("a b t1 t2", positive=True)
    b_complement = 1 - b
    a_complement = 1 - a
    E_naive = a * t1 / b + a_complement * t2 / b_complement
    truth = t1 + t2

    # Exactly unbiased at a==b, for arbitrary t1,t2.
    assert sp.simplify((E_naive - truth).subs(a, b)) == sp.Integer(0)

    # Genuinely biased at a concrete a!=b, t1!=t2 -- not just "generically
    # nonzero" in the abstract, a specific nonzero number.
    concrete_bias = (E_naive - truth).subs({a: sp.Rational(1, 4), b: sp.Rational(3, 4), t1: 2, t2: 5})
    assert sp.nsimplify(concrete_bias) != 0


def test_is_reweight_exactly_unbiased_when_support_holds_symbolic():
    a, t1, t2 = sp.symbols("a t1 t2", positive=True)
    # a in (0,1) strictly -- both species supported at the source.
    E_is_reweight = a * (t1 / a) + (1 - a) * (t2 / (1 - a))
    assert sp.simplify(E_is_reweight - (t1 + t2)) == sp.Integer(0)


def test_is_reweight_exact_bias_formula_under_support_violation_symbolic():
    t1, t2 = sp.symbols("t1 t2", positive=True)
    # a=0: species 2 (weight 1-a) is dead at the source, alive at the dest.
    # Only species 1 can ever be drawn; its score is t1/1 = t1.
    E_is_reweight = t1
    truth = t1 + t2
    relative_bias = sp.simplify((E_is_reweight - truth) / truth)
    assert sp.simplify(relative_bias - (-t2 / (t1 + t2))) == sp.Integer(0)


def test_fix_local_exactly_unbiased_and_independent_of_source_symbolic():
    a, b, t1, t2 = sp.symbols("a b t1 t2", positive=True)
    E_fix_local = b * (t1 / b) + (1 - b) * (t2 / (1 - b))
    assert sp.simplify(E_fix_local - (t1 + t2)) == sp.Integer(0)
    # A7: the expression must not depend on `a` (source vertex's proposal)
    # at all -- not merely "derivative zero", the symbol never appears.
    assert a not in sp.simplify(E_fix_local).free_symbols


def test_implementation_matches_support_violation_bias_formula_at_t10_config():
    # T10's own hard-cutoff config: species 0 dead above Z0, alive below;
    # candidates generated at Z_A (dead zone), reused at Z_B (alive zone).
    # Wide enough domain that the Gaussian absorption/emission tails (up to
    # +-6 sigma of the widest, EMISSION_SIGMA=70 at mu=590) are fully
    # captured by quadrature -- T10 itself uses a narrower [400,700] grid
    # and a loose 0.1 relative tolerance where truncation doesn't matter,
    # but this test's tight z-score check against an exact closed-form
    # prediction needs the quadrature truth to match the actual (untruncated,
    # continuous) sampling distribution to much higher precision.
    LAMBDA = torch.linspace(150.0, 1050.0, 6000)
    DLAM = (LAMBDA[1] - LAMBDA[0]).item()
    LAMBDA_S = 590.0
    MU = {0: 560.0, 1: 620.0}
    SIGMA = {0: 40.0, 1: 40.0}
    EMISSION_MU, EMISSION_SIGMA = 590.0, 70.0
    Z0 = 0.5
    Z_A, Z_B = 0.8, 0.2

    def _gaussian_pdf(x, mu, sigma):
        return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))

    A_TENSOR = {j: _gaussian_pdf(LAMBDA, MU[j], SIGMA[j]) for j in (0, 1)}
    LE_TENSOR = _gaussian_pdf(LAMBDA, EMISSION_MU, EMISSION_SIGMA)
    INTEGRAL_A_LE = {j: torch.trapz(A_TENSOR[j] * LE_TENSOR, LAMBDA).item() for j in (0, 1)}

    def absorption(j, lam):
        return _gaussian_pdf(torch.as_tensor(float(lam)), MU[j], SIGMA[j]).item()

    def emission(lam):
        return _gaussian_pdf(torch.as_tensor(float(lam)), EMISSION_MU, EMISSION_SIGMA).item()

    def excitation(j, lambda_s):
        return 1.0

    def conc_cutoff(j, z):
        if j == 0:
            return 1.5 if z < Z0 else 0.0
        return 1.0

    def _integral_a_le(j, z):
        return INTEGRAL_A_LE[j]

    def target_fn(y, z):
        j, lam = y
        return local_target(conc_cutoff, excitation, absorption, emission, j, lam, z, LAMBDA_S)

    def proposal_pdf_fn(y, z):
        j, lam = y
        weights = {k: species_weight(conc_cutoff, excitation, _integral_a_le, k, z, LAMBDA_S) for k in (0, 1)}
        total = weights[0] + weights[1]
        if total == 0.0:
            return 0.0
        return (weights[j] / total) * absorption(j, lam)

    def _mix(z):
        weights = {k: species_weight(conc_cutoff, excitation, _integral_a_le, k, z, LAMBDA_S) for k in (0, 1)}
        total = weights[0] + weights[1]
        return weights[0] / total, weights[1] / total

    # Per-species quadrature truth at z_B -- this file's t1 (species 1,
    # alive at z_A) / t2 (species 0, dead at z_A) role labels, matching the
    # symbolic toy's "t1 = surviving species, t2 = dead-at-source species".
    def _species_quadrature(j, z):
        total = 0.0
        for lam in LAMBDA:
            total += target_fn((j, lam.item()), z) * DLAM
        return total

    t_alive_at_A = _species_quadrature(1, Z_B)  # species 1: alive at both
    t_dead_at_A = _species_quadrature(0, Z_B)  # species 0: dead at Z_A
    truth_b = t_alive_at_A + t_dead_at_A
    predicted_relative_bias = -t_dead_at_A / truth_b  # this file's derived formula

    def _sample(z, rng):
        w0, _ = _mix(z)
        j = 0 if torch.rand((), generator=rng).item() < w0 else 1
        lam = torch.normal(MU[j], SIGMA[j], (1,), generator=rng).item()
        return j, lam

    N = 80_000
    rng = torch.Generator().manual_seed(22)
    samples = torch.empty(N)
    for t in range(N):
        y = _sample(Z_A, rng)
        s = is_reweight_score(target_fn, proposal_pdf_fn, y, Z_A, Z_B)
        samples[t] = 0.0 if s is None else s

    mean = samples.mean().item()
    stderr = samples.std().item() / (N ** 0.5)
    predicted_mean = truth_b * (1 + predicted_relative_bias)
    z = (mean - predicted_mean) / stderr
    assert abs(z) < 3.5, f"z={z}, mean={mean}, predicted={predicted_mean}"


if __name__ == "__main__":
    test_naive_exact_bias_formula_symbolic()
    test_is_reweight_exactly_unbiased_when_support_holds_symbolic()
    test_is_reweight_exact_bias_formula_under_support_violation_symbolic()
    test_fix_local_exactly_unbiased_and_independent_of_source_symbolic()
    test_implementation_matches_support_violation_bias_formula_at_t10_config()
    print("all A-heterogeneous-lookup-trichotomy tests passed")
