"""Symbolic (SymPy) proof of A9's Coverage Lemma disocclusion bias, in exact
closed form -- `restir_running_notes.md` section 12 states the Coverage
Lemma's necessary-and-sufficient support condition and reports it
"quadrature-verified exactly" against T18/Test 3a's measured -49.943% (vs.
predicted -50.000%), but the predicted number itself was only ever computed
numerically (quadrature), never derived as a closed-form algebraic
expression. This file supplies that derivation for T18's own shared-eval-
target disocclusion scenario (Test 3a: `gen_hist≡0`, `eval_hist==eval_cur`,
matching `test_t18_temporal_gen_eval_mismatch.py`'s exact setup) -- Test 3c's
number (different-shape eval densities, `sigma=8` vs `sigma=20`) is NOT
attempted here: that case's balance-heuristic weight varies with `x` (no
shared-target collapse to lean on), so its exact bias is a ratio-of-Gaussians
integral with no elementary closed form -- genuinely a quadrature-only
result, not a good SymPy target, unlike Test 3a.

**Setup, matching T18's `_run(g_gen=0.0, g_eval=g, ...)`:** the current
reservoir's gen-time target IS the shared destination target (`p_hat_cur,gen
== p_hat_d`, built the same frame). The history reservoir's gen-time target
was identically zero last frame (fully occluded) -- every streamed candidate
had weight `p_hat_hist,gen(x)/q(x) = 0/q(x) = 0`, so the reservoir never
accepts anything: `wsum_hist == 0` in every trial, deterministically, not
just in expectation. Both reservoirs are evaluated at the SAME shared
eval-time target `p_hat_d` (T18's `eval_target`, used for both `_target(g_eval)`
calls) -- exactly the shared-target-collapse precondition already proven in
`test_a_balance_heuristic_shared_target_collapse.py`.

**Derivation.**

1. Shared-target collapse (A8 specialization, already proven): since both
   reservoirs share one eval-time target and the shift between them is
   identity (same pixel/vertex domain), `m_cur(x) = c_cur/(c_cur+c_hist)`
   for every `x` -- constant, independent of shape/value.
2. History's term vanishes identically: `wsum_hist==0` in every trial means
   `m_hist(y_hist)*p_hat_d(y_hist)*W_hist` is exactly 0 sample-by-sample
   (not just in expectation), regardless of what `m_hist` or `y_hist`
   individually evaluate to.
3. Current's term is a plain RIS-unbiasedness instance with a CONSTANT
   multiplier pulled out front: `E[m_cur(y_cur)*p_hat_d(y_cur)*W_cur] =
   c_cur/(c_cur+c_hist) * E[p_hat_d(y_cur)*W_cur] = c_cur/(c_cur+c_hist) * TRUTH`
   (standard RIS unbiasedness, since `p_hat_cur,gen == p_hat_d` here).
4. Summing: `E[F] = TRUTH * c_cur/(c_cur+c_hist) + 0`, so the exact relative
   bias is `c_cur/(c_cur+c_hist) - 1 == -c_hist/(c_cur+c_hist)` -- collapsing
   to exactly `-1/2` (`-50.000%`) when `c_cur==c_hist`, matching T18's own
   `M_cur==M_hist==8` configuration precisely.

Test 1-2 mechanize this algebra symbolically for arbitrary confidences; test
3 cross-checks the derived closed form against the real `temporal_history.
temporal_combine` + `ris_reservoir.Reservoir` code at T18's own concrete
configuration (M=8 both sides), same continuous spectral scene as T18.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import sympy as sp
import torch

torch.set_default_dtype(torch.float64)

from ris_reservoir import Reservoir
from temporal_history import temporal_combine

from _temporal_reuse_common import (
    light_spectrum,
    gaussian,
    candidate_mass,
    gen_density,
    quadrature_truth,
)


def test_exact_relative_bias_formula_symbolic():
    # Steps 1-4 above, arbitrary symbolic positive confidences c_cur, c_hist
    # and an arbitrary nonzero symbolic TRUTH -- history's term is dropped
    # entirely (step 2: identically zero, not a variable), current's term
    # carries the shared-target collapse constant (step 1) times TRUTH
    # (step 3, plain RIS unbiasedness).
    c_cur, c_hist, TRUTH = sp.symbols("c_cur c_hist TRUTH", positive=True)

    m_cur_constant = c_cur / (c_cur + c_hist)  # A8 shared-target collapse
    E_F = m_cur_constant * TRUTH + 0  # history's term: identically 0 (step 2)

    relative_bias = sp.simplify((E_F - TRUTH) / TRUTH)
    expected = -c_hist / (c_cur + c_hist)
    assert sp.simplify(relative_bias - expected) == sp.Integer(0)


def test_relative_bias_collapses_to_exactly_minus_half_at_equal_confidence():
    c = sp.symbols("c", positive=True)  # c_cur == c_hist == c
    relative_bias = -c / (c + c)
    assert sp.simplify(relative_bias - sp.Rational(-1, 2)) == sp.Integer(0)


def test_implementation_matches_derivation_at_t18s_own_configuration():
    # Cross-check: real temporal_combine + Reservoir, T18's exact Test 3a
    # scene (continuous Gaussian target, M=8 candidates streamed per
    # reservoir, gen_hist≡0/eval shared), must land at the closed-form
    # predicted mean (TRUTH * c_cur/(c_cur+c_hist) = TRUTH/2 here) to within
    # ordinary MC noise -- not merely "biased in the right direction", but
    # numerically consistent with the exact -50.000% figure this file derived.
    L_E = light_spectrum()
    A = gaussian(550.0, 70.0)
    MASS = candidate_mass(L_E)
    P_GEN = gen_density(L_E)
    M = 8
    G_EVAL = 1.3

    truth = quadrature_truth(A * L_E * G_EVAL)
    predicted_mean = truth * sp.Rational(1, 2)  # c_cur==c_hist==M -> exactly 1/2

    def eval_target(idx):
        return (A[idx] * L_E[idx] * G_EVAL).item()

    def zero_target(idx):
        return 0.0

    def stream(target_fn, rng):
        r = Reservoir()
        for _ in range(M):
            idx = torch.multinomial(MASS, 1, generator=rng).item()
            p_hat = target_fn(idx)
            p_gen = P_GEN[idx].item()
            accepted = r.update(idx, p_hat / p_gen, rng)
            if accepted:
                r.set_p_hat_gen(p_hat)
        return r

    N = 20_000
    rng = torch.Generator().manual_seed(1901)
    samples = torch.empty(N)
    for t in range(N):
        r_cur = stream(eval_target, rng)  # gen ≡ eval, this frame
        r_hist = stream(zero_target, rng)  # gen≡0 last frame (fully occluded)
        combined = temporal_combine(r_cur, r_hist, eval_target, rng, wsum_gen_gate=-1.0)
        samples[t] = combined.wsum

    mean = samples.mean().item()
    stderr = samples.std().item() / (N ** 0.5)
    z = (mean - float(predicted_mean)) / stderr
    assert abs(z) < 3.5, f"z={z}, mean={mean}, predicted={float(predicted_mean)}"


if __name__ == "__main__":
    test_exact_relative_bias_formula_symbolic()
    test_relative_bias_collapses_to_exactly_minus_half_at_equal_confidence()
    test_implementation_matches_derivation_at_t18s_own_configuration()
    print("all A-coverage-lemma-disocclusion-exact-bias tests passed")
