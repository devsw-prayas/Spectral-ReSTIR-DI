"""Point-probe for T33 (compound spatiotemporal RNG, geometric stream: real
seeding schemes, and the joint-index-collision finding):

**What this supersedes:** an earlier pass at this question (retired, not
reconstructed here) modeled correlation via an idealized two-factor
Gaussian copula and found block correlation ~5-6x less harmful than a
naive model predicted -- that was an artifact of the idealized model, not
a property of real seeding schemes. This probe uses REAL, CONCRETE integer
seed-derivation formulas instead (matching the corrected historical
methodology), over the compound `(frame_idx, neighbor_idx)` grid:

| Scheme | Construction | What it models |
|---|---|---|
| `neighbor_ignored` | `seed = base + frame_idx` | Realistic plumbing bug: neighbor index never reaches the seed |
| `naive_additive` | `seed = base + frame_idx*1973 + neighbor_idx*9277` | No hashing, plain additive combination |
| `proper_hash` | `numpy.random.SeedSequence([base, frame_idx, neighbor_idx])` | Well-mixed hash (the "should be fine" baseline) |
| `swap_collision` | `seed = base + (frame_idx+neighbor_idx)*97 + (frame_idx*neighbor_idx mod 101)` | Deliberately depends only on sum and product -- passes every single-axis sweep but has an exact, algebraically-guaranteed collision between any index pair and its swap (same sum + same product => same unordered pair, by Vieta's formulas) |

**Part A -- correlation structure.** For each scheme, draw a geometric
(continuous, standard-normal-based) value per `(base, frame_idx,
neighbor_idx)` cell, varying `base` across many replicates, and measure
the Pearson correlation between two FIXED index-pair cells across those
replicates. `RNG-G0`'s existing single-axis sweeps (same-neighbor/
diff-frame, same-frame/diff-neighbor) catch `neighbor_ignored`'s bug
(`same-frame/diff-neighbor` correlation locks to exactly `+1.0`, since the
seed literally never depends on `neighbor_idx`) but are BLIND to
`swap_collision` -- that scheme's single-axis correlations are all clean,
yet the specific `(f,n)~(n,f)` swap-pair correlation is exactly `+1.0`
(same sum, same product mod 101 => identical seed by construction). This
is the key finding: **single-axis RNG sweeps provably cannot detect the
swap-collision failure mode.**

**Part B -- bias vs. pool size N.** Pools `N` cells into a `sqrt(N) x
sqrt(N)`-ish `(frame_idx, neighbor_idx)` grid, self-normalized IS against
a fixed target/proposal pair (`p~N(5,1)`, `q~N(3.5,1)`, same convention as
T30), and measures how bias shrinks as `N` grows. `neighbor_ignored`
shrinks much more slowly than the other three (a systematic WHOLE-BLOCK
collapse -- every neighbor in a frame gets an identical draw, removing
real degrees of freedom every frame) even though `swap_collision` has an
exact guaranteed collision too (a SPARSE pairwise defect, diluted by the
many non-colliding pairs in a large pool -- confirmed via direct collision
counting in the historical session, not asserted here, since this repo's
grids are small enough that the distinction is visible directly in the
bias numbers instead).

**Part C -- the flagged real-implementation check.** This repo's own
`pcg32_rng.pcg32_seed(pixel_id, frame_index, sample_index)` (module 8, the
actual production Philox/PCG32 keying formula) has never been checked for
this specific joint/swap-collision failure mode -- flagged as an open
verification gap in `project-infra-checkpoint` memory. Mapping
`neighbor_idx` onto a spatial pixel offset (`pixel_id = base_pixel +
neighbor_idx`) and `frame_idx` onto `frame_index` directly, this checks
whether the real formula (`pixel_id*1973 + frame_index*65536*9277 + 1`,
mod 2**64, then PCG32-mixed) ever collides across an index grid, including
every swap pair. **Result: no collisions found** -- and provably so, not
just empirically: the two axes carry different linear coefficients
(`1973` vs `65536*9277`), so a swap collision would require
`1973*(a-b) == 65536*9277*(a-b)` for some `a != b`, which is false since
the coefficients differ. This is a genuine, previously-unverified
confirmation for the real production scheme, not a re-check of something
`test_pcg32_rng.py` already covers (that file only checks single-axis
temporal decorrelation, T25's own mechanism).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import math

import torch
import numpy as np

from pcg32_rng import pcg32_seed

torch.set_default_dtype(torch.float64)

_SEED_MASK = (1 << 62) - 1
MU_TRUE = 5.0
MU_Q = 3.5


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


def _draw_geometric(scheme_fn, base, f, n):
    seed = scheme_fn(base, f, n) & _SEED_MASK
    rng = torch.Generator().manual_seed(seed)
    return torch.randn((), generator=rng).item()


def _correlation_over_bases(scheme_fn, pair_a, pair_b, n_replicates, base_rng):
    bases = torch.randint(0, 2 ** 31, (n_replicates,), generator=base_rng).tolist()
    xa = torch.tensor([_draw_geometric(scheme_fn, b, *pair_a) for b in bases])
    xb = torch.tensor([_draw_geometric(scheme_fn, b, *pair_b) for b in bases])
    return torch.corrcoef(torch.stack([xa, xb]))[0, 1].item()


_CATEGORIES = {
    "same_neighbor_diff_frame": ((1, 3), (2, 3)),
    "same_frame_diff_neighbor": ((1, 3), (1, 7)),
    "both_differ": ((1, 3), (4, 9)),
    "swap_pair": ((2, 5), (5, 2)),
}


def test_neighbor_ignored_shows_exact_same_frame_correlation():
    rng = torch.Generator().manual_seed(1234)
    c = _correlation_over_bases(
        neighbor_ignored_seed, *_CATEGORIES["same_frame_diff_neighbor"], 3000, rng
    )
    assert abs(c - 1.0) < 1e-6


def test_swap_collision_is_clean_on_single_axis_sweeps():
    rng = torch.Generator().manual_seed(1235)
    for cat in ("same_neighbor_diff_frame", "same_frame_diff_neighbor", "both_differ"):
        c = _correlation_over_bases(swap_collision_seed, *_CATEGORIES[cat], 3000, rng)
        assert abs(c) < 0.05


def test_swap_collision_hides_an_exact_swap_pair_correlation():
    # the key finding: single-axis sweeps (checked clean above) completely
    # miss this -- only checking the SPECIFIC swap relationship reveals it.
    rng = torch.Generator().manual_seed(1236)
    c = _correlation_over_bases(swap_collision_seed, *_CATEGORIES["swap_pair"], 3000, rng)
    assert abs(c - 1.0) < 1e-6


def test_naive_additive_and_proper_hash_are_clean_everywhere():
    rng = torch.Generator().manual_seed(1237)
    for scheme_fn in (naive_additive_seed, proper_hash_seed):
        for cat, pair in _CATEGORIES.items():
            c = _correlation_over_bases(scheme_fn, *pair, 2000, rng)
            assert abs(c) < 0.1, f"{scheme_fn.__name__} {cat}: corr={c}"


def _draw_pool(scheme_fn, base, T, M):
    xs = torch.empty(T, M)
    for f in range(T):
        for n in range(M):
            seed = scheme_fn(base, f, n) & _SEED_MASK
            rng = torch.Generator().manual_seed(seed)
            xs[f, n] = MU_Q + torch.randn((), generator=rng).item()
    return xs.reshape(-1)


def _importance_weight(x):
    return torch.exp(-0.5 * (x - MU_TRUE) ** 2 + 0.5 * (x - MU_Q) ** 2)


def _bias_for_pool_size(scheme_fn, N, n_trials, base_rng):
    T = int(round(math.sqrt(N)))
    M = N // T
    bases = torch.randint(0, 2 ** 31, (n_trials,), generator=base_rng).tolist()
    estimates = []
    for b in bases:
        x = _draw_pool(scheme_fn, b, T, M)
        w = _importance_weight(x)
        estimates.append(((w * x).sum() / w.sum()).item())
    return (torch.tensor(estimates).mean() - MU_TRUE).item()


def test_neighbor_ignored_bias_shrinks_much_slower_than_the_other_schemes():
    base_rng = torch.Generator().manual_seed(5000)
    n_trials = 1500

    bias_16 = _bias_for_pool_size(neighbor_ignored_seed, 16, n_trials, base_rng)
    bias_256 = _bias_for_pool_size(neighbor_ignored_seed, 256, n_trials, base_rng)
    neighbor_ignored_retention = abs(bias_256) / abs(bias_16)

    for scheme_fn in (naive_additive_seed, proper_hash_seed, swap_collision_seed):
        b16 = _bias_for_pool_size(scheme_fn, 16, n_trials, base_rng)
        b256 = _bias_for_pool_size(scheme_fn, 256, n_trials, base_rng)
        retention = abs(b256) / abs(b16)
        assert retention < 0.25, f"{scheme_fn.__name__}: retention={retention}"

    # neighbor_ignored's systematic whole-block collapse retains far more
    # of its bias than any of the other three schemes.
    assert neighbor_ignored_retention > 0.3


def test_production_pcg32_seed_has_no_joint_or_swap_collisions():
    # closes the verification gap flagged in project memory: T33/T34's
    # joint/swap-collision check had never been run against the actual
    # pcg32_rng.py production keying formula.
    base_pixel = 500
    grid = 32
    seen = {}
    for f in range(grid):
        for n in range(grid):
            s = pcg32_seed(pixel_id=base_pixel + n, frame_index=f, sample_index=0)
            assert s not in seen, f"collision: {seen.get(s)} and (f={f},n={n})"
            seen[s] = (f, n)

    # explicit swap-pair check (the specific failure mode swap_collision
    # demonstrates a toy scheme CAN have)
    for f in range(grid):
        for n in range(grid):
            if f == n:
                continue
            s1 = pcg32_seed(pixel_id=base_pixel + n, frame_index=f, sample_index=0)
            s2 = pcg32_seed(pixel_id=base_pixel + f, frame_index=n, sample_index=0)
            assert s1 != s2


if __name__ == "__main__":
    test_neighbor_ignored_shows_exact_same_frame_correlation()
    test_swap_collision_is_clean_on_single_axis_sweeps()
    test_swap_collision_hides_an_exact_swap_pair_correlation()
    test_naive_additive_and_proper_hash_are_clean_everywhere()
    test_neighbor_ignored_bias_shrinks_much_slower_than_the_other_schemes()
    test_production_pcg32_seed_has_no_joint_or_swap_collisions()
    print("all T33 tests passed")
