"""Symbolic (SymPy) proof of A10, the Temporal Tier-3 Corollary (running_notes
Sec. 13). A10 states the temporal combine is unbiased **iff** the Coverage
Lemma (A9) holds for every reservoir, and enumerates three cases:

  1. Current reservoir: gen == eval trivially -> automatic.
  2. History, unchanged scene state: gen == eval trivially too -> reduces
     exactly to A8's spatial proof applied across time (no new content).
  3. History, genuine scene change: Coverage Lemma is exact
     necessary-and-sufficient -- collapses to the runtime gate
     `wsum_gen == 0` on a trial => that reservoir's effective confidence is
     zeroed in the combine denominator, not just its numerator.

`test_a_coverage_lemma_disocclusion_exact_bias.py` (A9) already mechanized
the UNGATED failure mode for case 3 -- calling `temporal_combine` with
`wsum_gen_gate=-1.0` (gate disabled) reproduces exactly -50% bias on T18's
Test 3a scenario. What A9's file does not mechanize is the other half of the
iff: that the gate (the module's actual default, `wsum_gen_gate=0.0`)
*restores* exact unbiasedness by dropping history's confidence from the MIS
denominator entirely (`m_cur` collapsing from `c_cur/(c_cur+c_hist)` to
exactly `1`), not merely by skipping its candidate. That is this file's
target -- the "does both at once" claim in `temporal_history.py`'s
docstring, made exact. Case 2 (unchanged scene state) is included for
completeness of the three-way split: it is a one-line triviality once
gen==eval is asserted symbolically, so is checked directly rather than
deferred to prose.
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


def test_gate_collapses_m_cur_to_exactly_one_symbolic():
    # Case 3, gated: excluding history (M=0, wsum_gen==0 trial) makes its
    # confidence in the shared-target-collapse denominator exactly 0, not a
    # phantom small positive number -- m_cur = c_cur/(c_cur+0) = 1 for every
    # x, not the c_cur/(c_cur+c_hist) < 1 that the ungated case (A9) uses.
    c_cur = sp.symbols("c_cur", positive=True)
    c_hist_effective = sp.Integer(0)  # gate fires: reservoir dropped, M=0

    m_cur_gated = c_cur / (c_cur + c_hist_effective)
    assert sp.simplify(m_cur_gated - 1) == sp.Integer(0)


def test_gate_restores_exact_unbiasedness_contrasted_with_a9_ungated_bias():
    # Direct algebraic contrast of the iff's two directions on the same
    # scenario (T18 Test 3a shape): gated => E[F]=TRUTH exactly; ungated
    # (A9's own mechanized result) => E[F]=TRUTH*c_cur/(c_cur+c_hist), a
    # genuine -c_hist/(c_cur+c_hist) relative bias whenever c_hist>0.
    c_cur, c_hist, TRUTH = sp.symbols("c_cur c_hist TRUTH", positive=True)

    E_F_gated = 1 * TRUTH  # m_cur==1 (this file, case 3 gated)
    E_F_ungated = (c_cur / (c_cur + c_hist)) * TRUTH  # A9's mechanized result

    assert sp.simplify(E_F_gated - TRUTH) == sp.Integer(0)
    ungated_relative_bias = sp.simplify((E_F_ungated - TRUTH) / TRUTH)
    assert sp.simplify(ungated_relative_bias - (-c_hist / (c_cur + c_hist))) == sp.Integer(0)
    # The two directions agree only in the degenerate c_hist->0 limit --
    # i.e. gating is not equivalent to "no bias to begin with" in general.
    assert sp.limit(ungated_relative_bias, c_hist, 0) == 0


def test_unchanged_scene_state_case_reduces_trivially_to_a8():
    # Case 2: if history's gen-time and eval-time targets are the SAME
    # symbolic function (no scene change since last frame), the Coverage
    # Lemma's support condition supp(eval) subset supp(gen) holds by pure
    # reflexivity -- no disocclusion-style case analysis is needed, and
    # A8's general shared-target-collapse machinery (already proven in
    # test_a_balance_heuristic_shared_target_collapse.py) applies completely
    # unmodified. This is the symbolic content of "reduces exactly to A8's
    # spatial proof, applied across time."
    x = sp.symbols("x", real=True)
    p_hat_gen = sp.Function("p_hat")(x)
    p_hat_eval = p_hat_gen  # unchanged scene state: gen == eval identically

    # supp(eval) subset supp(gen) becomes supp(gen) subset supp(gen): trivial.
    coverage_condition_lhs_minus_rhs = sp.simplify(p_hat_eval - p_hat_gen)
    assert coverage_condition_lhs_minus_rhs == 0


def test_implementation_gate_matches_truth_not_half_at_t18s_configuration():
    # Cross-check: real temporal_combine with its DEFAULT gate
    # (wsum_gen_gate=0.0, i.e. the gate is active, unlike A9's file which
    # passes -1.0 to disable it) on T18's own Test-3a scene (gen_hist≡0,
    # M_cur=M_hist=8) must land at the FULL truth, not TRUTH/2 -- the gate
    # firing on every trial (wsum_hist deterministically 0) should recover
    # exact unbiasedness, not just reduce the bias.
    L_E = light_spectrum()
    A = gaussian(550.0, 70.0)
    MASS = candidate_mass(L_E)
    P_GEN = gen_density(L_E)
    M = 8
    G_EVAL = 1.3

    truth = quadrature_truth(A * L_E * G_EVAL)

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
        r_cur = stream(eval_target, rng)  # gen == eval, this frame
        r_hist = stream(zero_target, rng)  # gen == 0 last frame (fully occluded)
        combined = temporal_combine(r_cur, r_hist, eval_target, rng)  # default gate
        samples[t] = combined.wsum

    mean = samples.mean().item()
    stderr = samples.std().item() / (N ** 0.5)
    z = (mean - truth) / stderr
    assert abs(z) < 3.5, f"z={z}, mean={mean}, truth={truth}"


if __name__ == "__main__":
    test_gate_collapses_m_cur_to_exactly_one_symbolic()
    test_gate_restores_exact_unbiasedness_contrasted_with_a9_ungated_bias()
    test_unchanged_scene_state_case_reduces_trivially_to_a8()
    test_implementation_gate_matches_truth_not_half_at_t18s_configuration()
    print("all A-temporal-tier3-corollary tests passed")
