"""PCG32 RNG, production Philox/PCG32 keying scheme — checklist module 8.

Must match the actual production keying scheme, not plain numpy.random —
required for T25 (cross-frame RNG correlation check) and its real-tracer
counterpart V12.
Depends on: temporal_history (module 7).
"""

import torch


def pcg32_seed(pixel_id, frame_index, sample_index):
    """Derive the production per-draw PCG32/Philox key from (pixel, frame, sample)."""
    raise NotImplementedError


def pcg32_next(state):
    """Advance PCG32 state, return (next_state, uniform_float)."""
    raise NotImplementedError
