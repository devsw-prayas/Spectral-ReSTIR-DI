"""Point-probe for T25 (`forward_paper1_test_suite.md`: cross-frame RNG
correlation check -- does the production Philox/PCG32 keying scheme
decorrelate a pixel's current-frame draws from its own history reservoir's
draws). No A-item backs this: the gap was named in Session 9 planning and
never probed until module 8 (`pcg32_rng.py`) landed.

This is a DELIBERATE duplicate/promotion, not new coverage: module 8's own
`tests/test_pcg32_rng.py::test_t25_cross_frame_draws_are_uncorrelated`
already implements this exact mechanism and was treated as satisfying T25
at module-landing time (see `.claude/memory` project checkpoint). Every
other T-item gets its own dedicated `tests/test_t<N>_*.py` file per the
suite table's ID convention, so this file exists to give T25 the same
standalone home -- same computation, rewritten with this repo's `torch`
point-probe conventions (the original uses plain Python lists/loops since
`pcg32_rng.py` itself is scalar bit manipulation, not tensor ops) rather
than re-deriving a different check.

Mechanism: for each of many pixels, draw one "history" uniform at frame 0
(standing in for the draw that produced its history reservoir's candidate)
and one "current" uniform at frame 1 (this frame's fresh candidate draw),
same pixel both times, via the actual production
`pixelId*1973 + sampleIdx*9277 + 1` keying formula
(`pcg32_seed`/`pcg32_next`). If the keying scheme leaked frame-to-frame
structure, these two streams would show a detectable linear correlation
across pixels; Pearson |r| for two truly independent streams over N pixels
has stderr ~= 1/sqrt(N) ~= 0.0022 here, so 0.02 is a loose pass/fail band,
not a tight one.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from pcg32_rng import pcg32_seed, pcg32_next

torch.set_default_dtype(torch.float64)


def _draw_uniforms(frame_index, n):
    draws = torch.empty(n)
    for pixel_id in range(n):
        _, u = pcg32_next(pcg32_seed(pixel_id, frame_index=frame_index, sample_index=0))
        draws[pixel_id] = u
    return draws


def test_history_and_current_frame_draws_are_uncorrelated():
    N = 200_000
    hist = _draw_uniforms(frame_index=0, n=N)  # stands in for the history reservoir's draw
    cur = _draw_uniforms(frame_index=1, n=N)  # this frame's fresh candidate draw

    hist_c = hist - hist.mean()
    cur_c = cur - cur.mean()
    corr = (hist_c * cur_c).sum() / torch.sqrt((hist_c ** 2).sum() * (cur_c ** 2).sum())

    assert abs(corr.item()) < 0.02


def test_within_frame_draws_are_deterministic_and_distinct_across_pixels():
    # sanity companion to the correlation check above: the same (pixel,
    # frame) key must reproduce bit-exactly, and distinct pixels at the
    # same frame must not collide -- if either failed, the correlation
    # test above would be meaningless (either not testing what it claims,
    # or degenerate).
    N = 2_000
    draws_a = _draw_uniforms(frame_index=0, n=N)
    draws_b = _draw_uniforms(frame_index=0, n=N)
    assert torch.equal(draws_a, draws_b)
    assert len(torch.unique(draws_a)) == N


if __name__ == "__main__":
    test_history_and_current_frame_draws_are_uncorrelated()
    test_within_frame_draws_are_deterministic_and_distinct_across_pixels()
    print("all T25 tests passed")
