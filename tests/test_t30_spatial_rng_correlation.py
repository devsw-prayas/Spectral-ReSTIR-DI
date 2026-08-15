"""Point-probe for T30 (spatial RNG correlation confirmation).

Question: does the cross-frame RNG-correlation bias-floor pathology found
for the TEMPORAL axis (T25 -- correlated draws across frames of one pixel)
also apply on the SPATIAL axis -- correlated RNG streams across
neighboring pixels within one frame?

Direct analog of the temporal mechanism, pooling axis reinterpreted as `M`
spatial neighbors instead of `T` frames. Each neighbor proposes a candidate
from its own LOCAL target `q ~ N(mu_q=3.5, 1)`; the current pixel reweights
via the real self-normalized IS weight `w=p(x)/q(x)` against its OWN
target `p ~ N(mu_true=5, 1)` -- this is exactly the resampling-weight
structure `ris_reservoir.Reservoir` uses (`p_hat/p_gen`), simplified to
scalar Gaussians so the bias-floor mechanism is isolated from any other
module's machinery. Cross-neighbor correlation `rho` is injected via a
one-factor Gaussian copula (`x_i = mu_q + sqrt(rho)*z_common +
sqrt(1-rho)*z_i`, `z_common`/`z_i` iid standard normal) -- for normal
marginals this construction gives EXACT pairwise correlation `rho` between
neighbors directly (no separate copula-to-marginal transform needed, since
the marginal is already Gaussian). This models a weak-avalanche pixel-id
hash shared across a tile (e.g. a seeding scheme keyed off tile/Morton
index without proper hashing) -- the same failure class as T25/T33's
"naive seeding" concern, just on the spatial axis.

Estimator: self-normalized `sum(w_i * x_i) / sum(w_i)` over the `M`
neighbors, per trial; bias = mean estimate - `mu_true` over many trials.

**Expected pattern (matches T25's mechanism exactly, reconfirmed here on
the spatial axis, not re-derived):**
- `rho=0`: bias shrinks substantially as `M` grows (ordinary finite-sample
  self-normalized-IS bias, `O(1/M)`-ish decay).
- `rho>=0.5`: bias STALLS -- growing `M` from 32 to 128 barely moves it,
  because correlated neighbors don't add independent information no matter
  how many are pooled (`N_eff = M/(1+(M-1)*rho)` saturates at `1/rho` as
  `M -> infinity`).
- `rho=1.0`: fully frozen -- bias at every `M` matches the single-sample
  (`M=1`) bias almost exactly (all neighbors are literally the same draw).

No post-hoc statistical correction exists for this (same conclusion as
T25) -- the only fix is verified decorrelation at the seeding-scheme
level, which is exactly what T33/T34's real-scheme comparison checks.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

torch.set_default_dtype(torch.float64)

MU_TRUE = 5.0
MU_Q = 3.5


def _log_normal_pdf_unnorm(x, mu):
    # sigma=1 for both p and q; the 1/sqrt(2*pi) normalizer cancels in the
    # ratio below, so it's dropped rather than computed and canceled.
    return -0.5 * (x - mu) ** 2


def importance_weight(x):
    return torch.exp(_log_normal_pdf_unnorm(x, MU_TRUE) - _log_normal_pdf_unnorm(x, MU_Q))


def draw_correlated_neighbors(n_trials, M, rho, rng):
    """One-factor Gaussian copula, M neighbors per trial, exact pairwise
    correlation `rho` on normal marginals N(mu_q, 1)."""
    z_common = torch.randn(n_trials, 1, generator=rng)
    z_indiv = torch.randn(n_trials, M, generator=rng)
    z = (rho ** 0.5) * z_common + ((1 - rho) ** 0.5) * z_indiv
    return MU_Q + z


def measure_bias(rho, M, n_trials, rng):
    x = draw_correlated_neighbors(n_trials, M, rho, rng)
    w = importance_weight(x)
    est = (w * x).sum(dim=1) / w.sum(dim=1)
    return (est.mean() - MU_TRUE).item()


def test_neighbor_draws_have_the_intended_pairwise_correlation():
    rng = torch.Generator().manual_seed(3000)
    for rho in (0.0, 0.5, 0.9):
        x = draw_correlated_neighbors(200_000, 2, rho, rng)
        corr = torch.corrcoef(x.T)[0, 1].item()
        assert abs(corr - rho) < 0.02


def test_bias_decays_cleanly_with_pool_size_at_zero_correlation():
    rng = torch.Generator().manual_seed(3001)
    N = 200_000
    bias_1 = measure_bias(0.0, 1, N, rng)
    bias_128 = measure_bias(0.0, 128, N, rng)
    # ordinary self-normalized-IS finite-sample bias should shrink to a
    # small fraction of its M=1 value once pooling 128 INDEPENDENT
    # neighbors -- no floor when rho=0.
    assert abs(bias_128) < 0.1 * abs(bias_1)


def test_bias_stalls_at_high_correlation():
    rng = torch.Generator().manual_seed(3002)
    N = 200_000
    for rho in (0.9, 0.99):
        bias_1 = measure_bias(rho, 1, N, rng)
        bias_32 = measure_bias(rho, 32, N, rng)
        bias_128 = measure_bias(rho, 128, N, rng)
        # growing the pool from 32 to 128 barely moves the bias (the
        # non-vanishing floor) ...
        assert abs(bias_128 - bias_32) < 0.1 * abs(bias_1)
        # ... and the floor stays a large fraction of the M=1 bias, unlike
        # the clean rho=0 case which collapses toward zero.
        assert abs(bias_128) > 0.7 * abs(bias_1)


def test_bias_is_fully_frozen_at_perfect_correlation():
    rng = torch.Generator().manual_seed(3003)
    N = 200_000
    bias_1 = measure_bias(1.0, 1, N, rng)
    bias_128 = measure_bias(1.0, 128, N, rng)
    # rho=1: every "neighbor" is literally the same draw, so pooling more
    # of them changes nothing.
    assert abs(bias_128 - bias_1) < 0.05


if __name__ == "__main__":
    test_neighbor_draws_have_the_intended_pairwise_correlation()
    test_bias_decays_cleanly_with_pool_size_at_zero_correlation()
    test_bias_stalls_at_high_correlation()
    test_bias_is_fully_frozen_at_perfect_correlation()
    print("all T30 tests passed")
