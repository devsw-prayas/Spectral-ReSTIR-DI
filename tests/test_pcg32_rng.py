"""T-tier point-probes for module 8 (pcg32_rng.py).

Covers basic PCG32 correctness (determinism, 32-bit output range, uniformity)
and T25: whether the production `pixelId*1973 + sampleIdx*9277 + 1` keying
scheme (transcribed in pcg32_rng.py's module docstring) actually decorrelates
a pixel's current-frame draws from its own history reservoir's draws once
`frame_index` is folded into that key. Per forward_paper1_test_suite.md, T25
has no A-item backing it -- this is a genuine open empirical probe, not a
foregone-conclusion pass.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pcg32_rng import pcg32_seed, pcg32_next, MASK64, MASK32


def test_seed_and_next_are_deterministic():
    s1 = pcg32_seed(pixel_id=17, frame_index=3, sample_index=0)
    s2 = pcg32_seed(pixel_id=17, frame_index=3, sample_index=0)
    assert s1 == s2

    state = s1
    seq_a = []
    for _ in range(8):
        state, u = pcg32_next(state)
        seq_a.append(u)

    state = s2
    seq_b = []
    for _ in range(8):
        state, u = pcg32_next(state)
        seq_b.append(u)

    assert seq_a == seq_b


def test_seed_varies_with_each_key_component():
    base = pcg32_seed(0, 0, 0)
    diff_pixel = pcg32_seed(1, 0, 0)
    diff_frame = pcg32_seed(0, 1, 0)
    diff_sample = pcg32_seed(0, 0, 1)
    assert len({base, diff_pixel, diff_frame, diff_sample}) == 4


def test_state_and_output_stay_within_bit_widths():
    state = pcg32_seed(pixel_id=9999, frame_index=42, sample_index=7)
    assert 0 <= state <= MASK64
    for _ in range(1000):
        state, u = pcg32_next(state)
        assert 0 <= state <= MASK64
        assert 0.0 <= u < 1.0


def test_output_is_approximately_uniform():
    # No closed form for a single PRNG's empirical distribution -- Monte
    # Carlo mean/variance check against Uniform[0,1)'s true mean=0.5,
    # var=1/12, same tolerance discipline as the other modules' MC probes.
    N = 200_000
    state = pcg32_seed(pixel_id=1, frame_index=0, sample_index=0)
    draws = []
    for _ in range(N):
        state, u = pcg32_next(state)
        draws.append(u)

    mean = sum(draws) / N
    var = sum((u - mean) ** 2 for u in draws) / N
    assert abs(mean - 0.5) < 0.01
    assert abs(var - 1.0 / 12.0) < 0.01


def test_t25_cross_frame_draws_are_uncorrelated():
    # T25: for each of many pixels, draw one "history" uniform at frame 0
    # (standing in for the draw that produced its history reservoir's
    # candidate) and one "current" uniform at frame 1 (this frame's fresh
    # candidate draw), same pixel both times. If the keying scheme leaked
    # frame-to-frame structure, these two streams would show a detectable
    # linear correlation across pixels; Pearson |r| for two truly
    # independent streams over N pixels has stderr ~= 1/sqrt(N) ~= 0.0022
    # here, so 0.02 is a loose pass/fail band, not a tight one.
    N = 200_000
    hist = []
    cur = []
    for pixel_id in range(N):
        _, u_hist = pcg32_next(pcg32_seed(pixel_id, frame_index=0, sample_index=0))
        _, u_cur = pcg32_next(pcg32_seed(pixel_id, frame_index=1, sample_index=0))
        hist.append(u_hist)
        cur.append(u_cur)

    mean_h = sum(hist) / N
    mean_c = sum(cur) / N
    cov = sum((h - mean_h) * (c - mean_c) for h, c in zip(hist, cur)) / N
    var_h = sum((h - mean_h) ** 2 for h in hist) / N
    var_c = sum((c - mean_c) ** 2 for c in cur) / N
    corr = cov / (var_h * var_c) ** 0.5

    assert abs(corr) < 0.02


if __name__ == "__main__":
    test_seed_and_next_are_deterministic()
    test_seed_varies_with_each_key_component()
    test_state_and_output_stay_within_bit_widths()
    test_output_is_approximately_uniform()
    test_t25_cross_frame_draws_are_uncorrelated()
    print("all pcg32_rng tests passed")
