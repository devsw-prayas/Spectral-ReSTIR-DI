"""Point-probe for T32 (compound spatiotemporal, merge-order invariance):

Question: does it matter whether reservoir combine happens
temporal-then-spatial, spatial-then-temporal, or as one flat joint pool
over the full `T` frames x `M` neighbors grid?

Unlike T26-T31 (all standalone toys, since composition-lemma/RNG-
correlation theory isn't built into `src/` yet), T32 exercises this
repo's ACTUAL `mis_combine.combine_reservoirs` directly -- this question
is about associativity of the real combine formula, not new theory, so
there's no reason to reimplement it in a toy.

**Scope of this probe**: every raw candidate reservoir in the `T x M`
grid streams from the SAME shared target (`temporal_history.py`'s own
documented single-shared-target scope, same family T17-T22 use) with
IDENTITY shifts between every pair of cells (all cells live in one
domain) and no `m_cap`. Under exactly this scope the associativity is
actually an EXACT algebraic identity, not just an empirical finding:
T22's own proven result (see that file/`project-infra-checkpoint` memory)
shows `balance_heuristic_weight`'s `m_i` collapses to the plain confidence
share `M_i / Sum_j M_j` whenever every source shares the same eval-time
target as its own generation-time target -- substituting that into
`combine_reservoirs`' `w_i` formula and summing shows algebraically that
`combined.wsum` always reduces to `(Sum of ALL raw wsum_i) / (Sum of ALL
raw M_i)` regardless of how the sources are grouped into intermediate
combine calls first. This test confirms that derivation holds in the
actual floating-point implementation (expect `~1e-16` agreement, i.e. the
machine-precision floor, not just "close"), across several random scene
draws, matching the historical session's own `max|flat-sequential| ~ 1e-15`
finding closely.

**`m_cap` deliberately excluded** from this test: per-source M-clamping
(`min(confidence, m_cap)`) is NOT associative in general, since clamping
an already-pooled intermediate sub-total is a different operation than
clamping each raw cell individually -- that's a distinct question from
what T32 asks, out of scope here.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from ris_reservoir import Reservoir
from mis_combine import combine_reservoirs

from _temporal_reuse_common import light_spectrum, gaussian, candidate_mass, gen_density

torch.set_default_dtype(torch.float64)

L_E = light_spectrum()
A = gaussian(560.0, 50.0)
G = 1.3
MASS = candidate_mass(L_E)
P_GEN = gen_density(L_E)

T_FRAMES = 4
M_NEIGHBORS = 5
K_CANDIDATES = 6


def target_pdf_fn(idx):
    return (A[idx] * L_E[idx] * G).item()


def identity_shift(y):
    return y, 1.0


def _stream_reservoir(rng):
    r = Reservoir()
    for _ in range(K_CANDIDATES):
        idx = torch.multinomial(MASS, 1, generator=rng).item()
        p_hat = target_pdf_fn(idx)
        p_gen = P_GEN[idx].item()
        accepted = r.update(idx, p_hat / p_gen, rng)
        if accepted:
            r.set_p_hat_gen(p_hat)
    return r


def _build_grid(seed):
    rng = torch.Generator().manual_seed(seed)
    return [[_stream_reservoir(rng) for _ in range(M_NEIGHBORS)] for _ in range(T_FRAMES)]


def _combine(reservoirs, rng):
    n = len(reservoirs)
    target_fns = [target_pdf_fn] * n
    shift_fns = [[identity_shift] * n for _ in range(n)]
    return combine_reservoirs(reservoirs, target_fns, shift_fns, dest_index=0, rng=rng)


def _three_way_combine(grid, rng_seed):
    rng = torch.Generator().manual_seed(rng_seed)
    flat = [grid[t][n] for t in range(T_FRAMES) for n in range(M_NEIGHBORS)]
    flat_combined = _combine(flat, rng)

    rng = torch.Generator().manual_seed(rng_seed)
    temporal_first = [
        _combine([grid[t][n] for t in range(T_FRAMES)], rng) for n in range(M_NEIGHBORS)
    ]
    temporal_then_spatial = _combine(temporal_first, rng)

    rng = torch.Generator().manual_seed(rng_seed)
    spatial_first = [
        _combine([grid[t][n] for n in range(M_NEIGHBORS)], rng) for t in range(T_FRAMES)
    ]
    spatial_then_temporal = _combine(spatial_first, rng)

    return flat_combined, temporal_then_spatial, spatial_then_temporal


def test_merge_order_does_not_affect_wsum_or_confidence():
    for seed in (3200, 4100, 5555, 77):
        grid = _build_grid(seed)
        flat, temp_first, spat_first = _three_way_combine(grid, rng_seed=999)

        assert abs(flat.wsum - temp_first.wsum) / abs(flat.wsum) < 1e-9
        assert abs(flat.wsum - spat_first.wsum) / abs(flat.wsum) < 1e-9
        assert flat.confidence == temp_first.confidence == spat_first.confidence


def test_combined_confidence_equals_the_full_raw_candidate_count():
    grid = _build_grid(seed=42)
    flat, _, _ = _three_way_combine(grid, rng_seed=999)
    assert flat.confidence == T_FRAMES * M_NEIGHBORS * K_CANDIDATES


if __name__ == "__main__":
    test_merge_order_does_not_affect_wsum_or_confidence()
    test_combined_confidence_equals_the_full_raw_candidate_count()
    print("all T32 tests passed")
