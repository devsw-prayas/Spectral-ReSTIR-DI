"""PCG32 RNG, production Philox/PCG32 keying scheme -- checklist module 8.

Must match the actual production keying scheme, not plain numpy.random --
required for T25 (cross-frame RNG correlation check) and its real-tracer
counterpart V12.
Depends on: temporal_history (module 7).

This is a direct transcription of the production `PCG32` struct shared by
the sibling `FlashPath-FA2` / `WavefrontPathTracer` CUDA renderers
(`Core/Public/PCG32.cuh`): O'Neill's PCG32 (state' = state*MULT+INC,
XSH-RR output), fixed single stream (INC is a constant, not a per-instance
odd increment), advanced once on `seed()`.

Per-draw keying in that production code is
`seedValue = pixelId*1973 + sampleIdx*9277 + 1` (see
`Camera.cu::GeneratePrimaryRaysRGBKernel`) -- there is no `frameIndex` term
because that renderer is a plain per-frame path tracer: `sampleIdx` is
whatever monotonic sample ordinal the caller passes in, and a progressive
accumulator naturally passes an ever-increasing ordinal across frames. This
module's `pcg32_seed(pixel_id, frame_index, sample_index)` reproduces that
exact formula, folding `(frame_index, sample_index)` into that same single
ordinal slot the way a progressive caller would (`frame_index` and
`sample_index` occupy disjoint bit ranges so they can't alias each other):

    global_ordinal = frame_index * 65536 + sample_index
    seed_value     = pixel_id * 1973 + global_ordinal * 9277 + 1

The production formula was never designed with a reservoir's history
persisting *across* frames in mind, so whether this keying scheme actually
decorrelates a pixel's current-frame draws from its own history reservoir's
draws is an open empirical question, not a foregone-conclusion pass -- the
tests in this repo exist specifically to check it, not to assume it.
"""

MASK64 = (1 << 64) - 1
MASK32 = (1 << 32) - 1

PCG32_MULT = 6364136223846793005
PCG32_INC = 1442695040888963407

_PIXEL_STRIDE = 1973
_ORDINAL_STRIDE = 9277
_FRAME_STRIDE = 1 << 16  # disjoint bit range for sample_index within a frame


def pcg32_seed(pixel_id: int, frame_index: int, sample_index: int) -> int:
    """Derive the production per-draw PCG32 initial state from (pixel, frame, sample).

    Mirrors `PCG32::seed` (one state advance after the additive key), keyed
    by the production `pixelId*1973 + sampleIdx*9277 + 1` formula with
    `frame_index`/`sample_index` folded into a single monotonic ordinal
    (see module docstring).
    """
    global_ordinal = frame_index * _FRAME_STRIDE + sample_index
    seed_value = (pixel_id * _PIXEL_STRIDE + global_ordinal * _ORDINAL_STRIDE + 1) & MASK64

    state = (seed_value + PCG32_INC) & MASK64
    state = (state * PCG32_MULT + PCG32_INC) & MASK64
    return state


def pcg32_next(state: int) -> tuple[int, float]:
    """Advance PCG32 state, return (next_state, uniform_float) -- XSH-RR output."""
    state = (state * PCG32_MULT + PCG32_INC) & MASK64

    xorshifted = (((state >> 18) ^ state) >> 27) & MASK32
    rot = (state >> 59) & 0x1F

    out = ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & MASK32

    return state, out * (1.0 / 4294967296.0)
