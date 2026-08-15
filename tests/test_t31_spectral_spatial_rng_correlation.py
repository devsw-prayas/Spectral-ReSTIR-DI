"""Point-probe for T31 (spectral x spatial RNG correlation, fills the last
cell of the original 2x2): does the RNG bias-floor pathology (T25 temporal,
T30 spatial, both on the GEOMETRIC stream) also hit the SPECTRAL stream
pooled across spatial neighbors?

Two-species mixture, standing in for rank-k species selection (same
`heterogeneous_lookup`-style rank-2 flavor as T7-T11, but here isolating
the RNG-correlation question specifically, not the reuse-strategy
question): shared component means `[480, 620]`, shared `sigma=15`,
PROPOSAL species weights `[0.5, 0.5]` (each neighbor's local candidate
generation) vs. TARGET species weights `[0.2, 0.8]` (the true, more
species-2-heavy distribution) -- self-normalized IS weight
`w(lambda) = p(lambda)/q(lambda)` corrects for the shape mismatch, exactly
like `heterogeneous_lookup.py`'s IS-reweight strategy, simplified to a
scalar 2-component mixture to isolate the RNG-correlation mechanism from
any position/context-dependence.

**What's new relative to T30 (spatial, geometric stream):** T30 injects
correlation on a single continuous draw per neighbor. Here TWO separate
draws are correlated per neighbor, via TWO independent one-factor Gaussian
copulas at the SAME strength `rho` -- (1) the DISCRETE species-selection
draw (`species_i = 1 if z_species_i > 0 else 0`, a median split exactly
matching the proposal's 50/50 weights) and (2) the CONTINUOUS wavelength
quantile within whichever species was selected
(`lambda_i = mu[species_i] + sigma * z_pos_i`). Using two independent
common factors (not one shared factor for both) keeps "correlated which
species gets picked" and "correlated where within that species" as
genuinely separate failure modes, matching this addendum's own framing
("correlated species-selection AND wavelength quantiles").

**Expected pattern (same non-vanishing bias floor as every other cell of
the stream x axis-structure matrix -- T25, T30 -- confirmed here, not
re-derived):** clean `O(1/M)`-ish bias decay at `rho=0`; a stalled floor
by `M=32` already at `rho=0.5` (the addendum's own specific finding, a
faster stall than T30's spatial-geometric cell needed rho>=0.9 for) --
reconfirmed with fresh parameters below.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

torch.set_default_dtype(torch.float64)

MU = torch.tensor([480.0, 620.0])
SIGMA = 15.0
Q_WEIGHTS = torch.tensor([0.5, 0.5])  # proposal (candidate-generation) species weights
P_WEIGHTS = torch.tensor([0.2, 0.8])  # true target species weights

TRUTH = (P_WEIGHTS * MU).sum().item()


def _normal_pdf(x, mu, sigma):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * (2 * torch.pi) ** 0.5)


def _mixture_pdf(x, weights):
    return weights[0] * _normal_pdf(x, MU[0], SIGMA) + weights[1] * _normal_pdf(x, MU[1], SIGMA)


def draw_correlated_pool(n_trials, M, rho, rng):
    """M spatial neighbors per trial; species-selection and wavelength
    quantile each independently correlated across neighbors at strength rho."""
    zc_species = torch.randn(n_trials, 1, generator=rng)
    zi_species = torch.randn(n_trials, M, generator=rng)
    z_species = (rho ** 0.5) * zc_species + ((1 - rho) ** 0.5) * zi_species

    zc_pos = torch.randn(n_trials, 1, generator=rng)
    zi_pos = torch.randn(n_trials, M, generator=rng)
    z_pos = (rho ** 0.5) * zc_pos + ((1 - rho) ** 0.5) * zi_pos

    species = (z_species > 0).long()  # median split == 50/50, matches Q_WEIGHTS
    mu_sel = MU[species]
    lam = mu_sel + SIGMA * z_pos
    return lam


def measure_bias(rho, M, n_trials, rng):
    lam = draw_correlated_pool(n_trials, M, rho, rng)
    w = _mixture_pdf(lam, P_WEIGHTS) / _mixture_pdf(lam, Q_WEIGHTS)
    est = (w * lam).sum(dim=1) / w.sum(dim=1)
    return (est.mean() - TRUTH).item()


def test_species_selection_matches_the_intended_proposal_split():
    rng = torch.Generator().manual_seed(3100)
    lam = draw_correlated_pool(200_000, 1, 0.0, rng)
    frac_species_2 = (lam > (MU[0] + MU[1]) / 2).float().mean().item()
    assert abs(frac_species_2 - 0.5) < 0.01


def test_bias_decays_cleanly_with_pool_size_at_zero_correlation():
    rng = torch.Generator().manual_seed(3101)
    N = 200_000
    bias_1 = measure_bias(0.0, 1, N, rng)
    bias_128 = measure_bias(0.0, 128, N, rng)
    assert abs(bias_128) < 0.1 * abs(bias_1)


def test_bias_stalls_by_m32_already_at_rho_0_5():
    # the addendum's own specific finding: unlike T30's geometric-spatial
    # cell (needed rho>=0.9 to stall), this cell stalls at a much weaker
    # rho=0.5 already by M=32.
    rng = torch.Generator().manual_seed(3102)
    N = 200_000
    bias_1 = measure_bias(0.5, 1, N, rng)
    bias_32 = measure_bias(0.5, 32, N, rng)
    bias_128 = measure_bias(0.5, 128, N, rng)
    assert abs(bias_128 - bias_32) < 0.15 * abs(bias_1)
    assert abs(bias_128) > 0.2 * abs(bias_1)


def test_bias_stalls_harder_at_high_correlation():
    rng = torch.Generator().manual_seed(3103)
    N = 200_000
    for rho in (0.9, 0.99):
        bias_1 = measure_bias(rho, 1, N, rng)
        bias_32 = measure_bias(rho, 32, N, rng)
        bias_128 = measure_bias(rho, 128, N, rng)
        assert abs(bias_128 - bias_32) < 0.1 * abs(bias_1)
        assert abs(bias_128) > 0.6 * abs(bias_1)


if __name__ == "__main__":
    test_species_selection_matches_the_intended_proposal_split()
    test_bias_decays_cleanly_with_pool_size_at_zero_correlation()
    test_bias_stalls_by_m32_already_at_rho_0_5()
    test_bias_stalls_harder_at_high_correlation()
    print("all T31 tests passed")
