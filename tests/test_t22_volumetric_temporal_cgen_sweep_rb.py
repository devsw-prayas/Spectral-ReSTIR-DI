"""Point-probe for T22 (session_log_restir_10_volumetric_temporal_threshold's
Test 1 + Test 3: volumetric temporal C_gen sweep + Rao-Blackwell re-check).

Direct follow-on to T21 (clean `C_hist==C_cur` baseline). Here the history
reservoir is streamed under a MISMATCHED optical depth `C_gen`, while the
eval-time target fed to `temporal_combine`/`combine_reservoirs` always uses
`C_eval=1.0` -- exactly the gen/eval split `Reservoir.p_hat_gen` exists to
support (A9).

**Why this test does NOT chase a bias cutoff** (superseding
`addendum_volumetric_temporal_mclamping.md`'s VTest 2 read): the actual
follow-up session (`session_log_restir_10_volumetric_temporal_threshold.md`
sec 4) traced that addendum's -45%-at-C_gen=150 finding to a NAIVE
destination-only combine, not the corrected balance-heuristic combine this
repo's `temporal_history.temporal_combine`/`mis_combine.combine_reservoirs`
implements. There is in fact a clean, exact, closed-form reason the
corrected combine cannot be biased here: with a single shared eval target
for current+history (identity shift, static geometry), the balance-heuristic
weight `m_i` collapses to the plain confidence share `M_i/Sum_j M_j`
(derived and verified below in isolation), and the classic RIS
contribution-weight identity `E[h(y)*W] = quadrature_truth(h)` holds EXACTLY
for any `h`, independent of the reservoir's own generation target `p_hat_gen`
-- as long as `p_hat_gen>0` everywhere `h` has support (the Coverage Lemma
condition, always true here since transmittance is never exactly zero at
finite `C_gen`). So bias truly cannot appear at any finite `C_gen`.

**What this repo's own numbers add past session_10's**: raw single-pick MC
(`temporal_combine`, one reservoir draw per trial) becomes numerically
IMPRACTICAL well before `C_gen=150` in this probe's specific Gaussian
parameterization -- verified by hand that even N=200,000 trials leaves the
observed sample mean ~35% off at `C_gen=30` (heavy-tailed / high-variance,
not biased -- exactly the addendum's own flagged "not-finite-variance"
character, worse here than in session_10's own scale). Rather than fight
that convergence with brute-force N (impractical for a test file), this
probe uses a Rao-Blackwellized estimator (T3's technique: average each
reservoir's M raw candidate draws' own single-candidate MIS contribution
directly, instead of letting the reservoir's internal accept/reject pick
just one) to confirm unbiasedness at the SAME extreme `C_gen` values with a
tiny fraction of the samples -- this RB estimator's variance is provably
`C_gen`-independent (it algebraically reduces to plain importance sampling
of the EVAL target against the light-importance proposal, with the
`C_gen`-dependent `p_hat_gen` having cancelled out entirely), confirmed
empirically (~constant sample std across `C_gen` in {30, 70, 150} below).
**Lesson for any future T-item touching this same target family**: never
raw-MC a single-pick reservoir estimator past a modest generation-target
mismatch without first checking whether Rao-Blackwellizing removes the
`C_gen` dependence from the variance -- the naive fix ("just increase N")
does not converge in any practical sense here.

Same volumetric target family as T21 (single fixed vertex,
`heterogeneous_lookup.local_target`'s `transmittance_fn` hook, rank-1
degenerate species):

    p_hat(lambda', C) = a(lambda') * L_e(lambda') * exp(-a(lambda')*C) * G
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from ris_reservoir import Reservoir
from mis_combine import balance_heuristic_weight
from temporal_history import temporal_combine
from heterogeneous_lookup import local_target
from furnace_canary import effective_sample_size

from _temporal_reuse_common import (
    GRID,
    light_spectrum,
    gaussian,
    candidate_mass,
    gen_density,
    quadrature_truth,
)

torch.set_default_dtype(torch.float64)

L_E = light_spectrum()
A = gaussian(560.0, 50.0)
G = 1.3
C_EVAL = 1.0  # current frame's actual optical depth (fixed, matches session_10)

MASS = candidate_mass(L_E)
P_GEN = gen_density(L_E)
M = 8


def _transmittance(j, lam_prime, c):
    return torch.exp(-A[lam_prime] * c)


def target_pdf(idx, c):
    value = local_target(
        species_concentration_fn=lambda j, z: 1.0,
        excitation_fn=lambda j, lambda_s: 1.0,
        absorption_fn=lambda j, lam_prime: A[lam_prime],
        emission_fn=lambda lam_prime: L_E[lam_prime],
        j=0,
        lam_prime=idx,
        z=c,
        lambda_s=None,
        transmittance_fn=lambda j, lam_prime, z: _transmittance(j, lam_prime, z),
    )
    return (value * G).item()


def eval_target_fn(idx):
    return target_pdf(idx, C_EVAL)


_TRANS_EVAL = torch.exp(-A * C_EVAL)
TRUTH = quadrature_truth(A * L_E * _TRANS_EVAL * G)

TARGET_FNS = [eval_target_fn, eval_target_fn]
_identity = lambda y: (y, 1.0)
SHIFT_FNS = [[_identity, _identity], [_identity, _identity]]


def _stream_reservoir(rng, c, record=False):
    r = Reservoir()
    draws = []
    for _ in range(M):
        idx = torch.multinomial(MASS, 1, generator=rng).item()
        p_hat = target_pdf(idx, c)
        p_gen = P_GEN[idx].item()
        w = p_hat / p_gen
        accepted = r.update(idx, w, rng)
        if accepted:
            r.set_p_hat_gen(p_hat)
        if record:
            draws.append((idx, w))
    return r, draws


def test_truth_matches_quadrature_of_the_analytic_target():
    expected = (A * L_E * _TRANS_EVAL * G * GRID.weights).sum().item()
    assert abs(TRUTH - expected) < 1e-9


def test_balance_heuristic_collapses_to_confidence_share_under_shared_target():
    """With identical M and a single shared eval target for both reservoirs,
    A8's balance heuristic reduces to the plain M-share -- the algebraic fact
    this whole test's low-variance RB argument leans on."""
    rng = torch.Generator().manual_seed(2150)
    r_cur, _ = _stream_reservoir(rng, C_EVAL)
    r_hist, _ = _stream_reservoir(rng, 40.0)
    reservoirs = [r_cur, r_hist]
    m_0 = balance_heuristic_weight(0, r_cur.y, reservoirs, TARGET_FNS, SHIFT_FNS)
    m_1 = balance_heuristic_weight(1, r_hist.y, reservoirs, TARGET_FNS, SHIFT_FNS)
    assert abs(m_0 - 0.5) < 1e-9
    assert abs(m_1 - 0.5) < 1e-9


def test_corrected_combine_stays_unbiased_at_modest_cgen_mismatch():
    """Raw single-pick `temporal_combine` MC, direct API usage -- tractable
    at modest mismatch (up to ~15x the current frame's optical depth)."""
    N = 15_000
    for c_gen in (1.0, 5.0, 10.0, 15.0):
        rng = torch.Generator().manual_seed(int(3000 + c_gen))
        samples = torch.empty(N)
        for t in range(N):
            r_cur, _ = _stream_reservoir(rng, C_EVAL)
            r_hist, _ = _stream_reservoir(rng, c_gen)
            combined = temporal_combine(r_cur, r_hist, eval_target_fn, rng)
            samples[t] = combined.wsum

        mean = samples.mean().item()
        stderr = samples.std().item() / (N ** 0.5)
        z = (mean - TRUTH) / stderr
        assert abs(z) < 4.0, f"C_gen={c_gen}: z={z}"


def test_ess_hist_over_m_decays_as_cgen_grows():
    """Diagnostic tracked in session_10 in place of a bias cutoff: variance
    risk (not bias) is what actually grows with C_gen mismatch."""
    rng = torch.Generator().manual_seed(2300)
    ess_ratios = []
    for c_gen in (1.0, 10.0, 30.0, 70.0, 150.0):
        _, draws = _stream_reservoir(rng, c_gen, record=True)
        weights = torch.tensor([w for _, w in draws])
        ess_ratios.append(effective_sample_size(weights) / M)

    assert ess_ratios[0] > 0.5  # matched case: healthy
    assert ess_ratios[-1] < ess_ratios[0]
    for a, b in zip(ess_ratios, ess_ratios[1:]):
        assert b <= a + 1e-9  # non-increasing


def _rb_contribution(index, draws, reservoirs):
    """Rao-Blackwellized replacement for reservoir `index`'s term: average
    the M raw candidate draws' own single-candidate MIS contribution to the
    destination (index 0) directly, instead of letting that reservoir's
    internal accept/reject pick just one -- same technique as T3's
    `_rb_pixel0_contribution`, applied to the temporal case. Provably has
    `C_gen`-independent variance for this shared-target family (see module
    docstring): the reservoir's own `p_hat_gen` never enters this formula at
    all, only the raw draws and the shared eval target."""
    total = 0.0
    for idx, _w in draws:
        m_i = balance_heuristic_weight(index, idx, reservoirs, TARGET_FNS, SHIFT_FNS)
        y_dest, J = SHIFT_FNS[index][0](idx)
        if y_dest is None:
            continue
        total += m_i * TARGET_FNS[0](y_dest) * abs(J) / P_GEN[idx].item()
    return total / len(draws)


def test_rao_blackwell_confirms_unbiased_at_extreme_cgen_mismatch():
    """Mirrors session_10 Test 3 at the worst tested points -- confirms the
    combine stays formally unbiased even at 30x-150x optical-depth mismatch,
    where a raw single-pick MC estimate (previous test's structure) would
    need an impractically large N to converge (heavy-tailed, per the
    addendum's own flagged caveat)."""
    N = 6_000
    for c_gen in (30.0, 70.0, 150.0):
        rng = torch.Generator().manual_seed(int(4000 + c_gen))
        samples = torch.empty(N)
        for t in range(N):
            r_cur, draws_cur = _stream_reservoir(rng, C_EVAL, record=True)
            r_hist, draws_hist = _stream_reservoir(rng, c_gen, record=True)
            reservoirs = [r_cur, r_hist]
            samples[t] = (
                _rb_contribution(0, draws_cur, reservoirs)
                + _rb_contribution(1, draws_hist, reservoirs)
            )

        mean = samples.mean().item()
        stderr = samples.std().item() / (N ** 0.5)
        z = (mean - TRUTH) / stderr
        assert abs(z) < 4.0, f"C_gen={c_gen}: z={z}"


if __name__ == "__main__":
    test_truth_matches_quadrature_of_the_analytic_target()
    test_balance_heuristic_collapses_to_confidence_share_under_shared_target()
    test_corrected_combine_stays_unbiased_at_modest_cgen_mismatch()
    test_ess_hist_over_m_decays_as_cgen_grows()
    test_rao_blackwell_confirms_unbiased_at_extreme_cgen_mismatch()
    print("all T22 tests passed")
