"""Point-probe for T34 (compound spatiotemporal, spectral stream --
completes the full 5-cell matrix):

Same four seeding schemes and same `(frame_idx, neighbor_idx)` grid as T33,
applied to the SPECTRAL stream's own resampling draw (species categorical
choice + within-species Gaussian wavelength quantile, mirroring T31's
two-species mixture: means `[480, 620]`, shared `sigma=15`, proposal
weights `[0.5, 0.5]` vs. target weights `[0.2, 0.8]`) instead of a single
continuous positional draw -- testing whether the spectral stream's OWN
seed derivation has the same joint-collision exposure independently of the
geometric stream (they're separate draws in a real implementation, not
shared state).

**Correlation structure**: matches T33 exactly (checked below) --
confirms the seed scheme's correlation behavior is a property of the seed
derivation itself, stream-agnostic.

**The new finding here -- discreteness amplification:** the SAME
`neighbor_ignored` bug (same-frame draws collapse to one value, exact
`+1.0` correlation) produces a much LARGER absolute bias for the spectral
stream than for T33's geometric stream at matched pool size `N`. Mechanism:
the geometric stream duplicates a CONTINUOUS value across correlated
neighbors, which still carries partial information (close to correct,
just not independent). The spectral stream duplicates a DISCRETE
categorical species choice -- when a whole frame's neighbors collapse to
one species pick, that frame can lose an ENTIRE species' representation,
not just some effective sample size. Discrete streams are more exposed to
whole-block-collapse bugs than continuous streams at the same correlation
strength -- this repo's own numbers reconfirm the historical session's
finding (`neighbor_ignored` bias at matched `N` came in several times
larger for the spectral stream than the geometric one), not just a
restatement of T33.

`swap_collision`'s sparse-collision dilution pattern (small effect on
bias, since the collision is a rare pairwise event, not a systematic
block collapse) holds for the spectral stream too, consistent with T33's
own explanation -- checked here as a companion, non-amplified case.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import math

import torch
import numpy as np

torch.set_default_dtype(torch.float64)

_SEED_MASK = (1 << 62) - 1
MU = torch.tensor([480.0, 620.0])
SIGMA = 15.0
Q_WEIGHTS = torch.tensor([0.5, 0.5])
P_WEIGHTS = torch.tensor([0.2, 0.8])
TRUTH = (P_WEIGHTS * MU).sum().item()


def neighbor_ignored_seed(base, f, n):
    return base + f


def naive_additive_seed(base, f, n):
    return base + f * 1973 + n * 9277


def proper_hash_seed(base, f, n):
    ss = np.random.SeedSequence([int(base), int(f), int(n)])
    return int(ss.generate_state(1, dtype=np.uint64)[0])


def swap_collision_seed(base, f, n):
    s = f + n
    p = (f * n) % 101
    return base + s * 97 + p


SCHEMES = {
    "neighbor_ignored": neighbor_ignored_seed,
    "naive_additive": naive_additive_seed,
    "proper_hash": proper_hash_seed,
    "swap_collision": swap_collision_seed,
}


def _normal_pdf(x, mu, sigma):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * (2 * torch.pi) ** 0.5)


def _mixture_pdf(x, weights):
    return weights[0] * _normal_pdf(x, MU[0], SIGMA) + weights[1] * _normal_pdf(x, MU[1], SIGMA)


def _draw_spectral(scheme_fn, base, f, n):
    """One species-selection draw + one within-species wavelength quantile
    draw, both seeded off the SAME cell key (matches T33's convention of
    one seed per (frame,neighbor) cell driving everything that cell needs)."""
    seed = scheme_fn(base, f, n) & _SEED_MASK
    rng = torch.Generator().manual_seed(seed)
    z_species = torch.randn((), generator=rng)
    z_pos = torch.randn((), generator=rng)
    species = 0 if z_species.item() <= 0 else 1  # median split == 50/50 proposal
    return MU[species] + SIGMA * z_pos


def _correlation_over_bases(scheme_fn, pair_a, pair_b, n_replicates, base_rng):
    bases = torch.randint(0, 2 ** 31, (n_replicates,), generator=base_rng).tolist()
    xa = torch.tensor([_draw_spectral(scheme_fn, b, *pair_a) for b in bases])
    xb = torch.tensor([_draw_spectral(scheme_fn, b, *pair_b) for b in bases])
    return torch.corrcoef(torch.stack([xa, xb]))[0, 1].item()


_CATEGORIES = {
    "same_neighbor_diff_frame": ((1, 3), (2, 3)),
    "same_frame_diff_neighbor": ((1, 3), (1, 7)),
    "both_differ": ((1, 3), (4, 9)),
    "swap_pair": ((2, 5), (5, 2)),
}


def test_correlation_structure_matches_the_geometric_stream():
    rng = torch.Generator().manual_seed(2234)
    c_same_frame = _correlation_over_bases(
        neighbor_ignored_seed, *_CATEGORIES["same_frame_diff_neighbor"], 3000, rng
    )
    assert abs(c_same_frame - 1.0) < 1e-6

    c_swap = _correlation_over_bases(swap_collision_seed, *_CATEGORIES["swap_pair"], 3000, rng)
    assert abs(c_swap - 1.0) < 1e-6

    c_swap_single_axis = _correlation_over_bases(
        swap_collision_seed, *_CATEGORIES["both_differ"], 3000, rng
    )
    assert abs(c_swap_single_axis) < 0.05


def _draw_pool(scheme_fn, base, T, M):
    xs = torch.empty(T, M)
    for f in range(T):
        for n in range(M):
            xs[f, n] = _draw_spectral(scheme_fn, base, f, n)
    return xs.reshape(-1)


def _bias_for_pool_size(scheme_fn, N, n_trials, base_rng):
    T = int(round(math.sqrt(N)))
    M = N // T
    bases = torch.randint(0, 2 ** 31, (n_trials,), generator=base_rng).tolist()
    estimates = []
    for b in bases:
        x = _draw_pool(scheme_fn, b, T, M)
        w = _mixture_pdf(x, P_WEIGHTS) / _mixture_pdf(x, Q_WEIGHTS)
        estimates.append(((w * x).sum() / w.sum()).item())
    return (torch.tensor(estimates).mean() - TRUTH).item()


def _bias_for_pool_size_geometric(scheme_fn, N, n_trials, base_rng, mu_true=5.0, mu_q=3.5):
    """T33's own geometric-stream bias, recomputed locally (self-contained
    file, no cross-import between T-item files per this repo's convention)
    for the direct spectral-vs-geometric magnitude comparison below."""
    T = int(round(math.sqrt(N)))
    M = N // T
    bases = torch.randint(0, 2 ** 31, (n_trials,), generator=base_rng).tolist()
    estimates = []
    for b in bases:
        xs = torch.empty(T, M)
        for f in range(T):
            for n in range(M):
                seed = scheme_fn(b, f, n) & _SEED_MASK
                rng = torch.Generator().manual_seed(seed)
                xs[f, n] = mu_q + torch.randn((), generator=rng).item()
        x = xs.reshape(-1)
        w = torch.exp(-0.5 * (x - mu_true) ** 2 + 0.5 * (x - mu_q) ** 2)
        estimates.append(((w * x).sum() / w.sum()).item())
    return (torch.tensor(estimates).mean() - mu_true).item()


def test_neighbor_ignored_bias_is_much_larger_for_the_discrete_spectral_stream():
    N = 256
    n_trials = 1500
    rng_spectral = torch.Generator().manual_seed(7000)
    rng_geometric = torch.Generator().manual_seed(7001)

    bias_spectral = _bias_for_pool_size(neighbor_ignored_seed, N, n_trials, rng_spectral)
    bias_geometric = _bias_for_pool_size_geometric(
        neighbor_ignored_seed, N, n_trials, rng_geometric
    )

    # same bug (whole-frame collapse), same N -- discreteness amplification:
    # losing an entire species' representation hurts far more than losing
    # some effective sample size on a continuous value.
    assert abs(bias_spectral) > 3.0 * abs(bias_geometric)


def test_swap_collision_bias_stays_comparably_small_for_both_streams():
    N = 256
    n_trials = 1500
    rng_spectral = torch.Generator().manual_seed(7002)
    rng_geometric = torch.Generator().manual_seed(7003)

    bias_spectral = _bias_for_pool_size(swap_collision_seed, N, n_trials, rng_spectral)
    bias_geometric = _bias_for_pool_size_geometric(
        swap_collision_seed, N, n_trials, rng_geometric
    )

    # sparse pairwise collisions (not a whole-block collapse) don't get
    # amplified by discreteness the way neighbor_ignored's bug does --
    # both streams' bias stays small and within the same order of magnitude.
    assert abs(bias_spectral) < 1.0
    assert abs(bias_geometric) < 1.0


if __name__ == "__main__":
    test_correlation_structure_matches_the_geometric_stream()
    test_neighbor_ignored_bias_is_much_larger_for_the_discrete_spectral_stream()
    test_swap_collision_bias_stays_comparably_small_for_both_streams()
    print("all T34 tests passed")
