"""Temporal history buffer + combine rule — checklist module 7.

Coverage Lemma (A9, restir_running_notes.md §11) and Temporal Tier-3
Corollary (A10, §12): under static geometry (reprojection Jacobian ≡ 1,
the locked v1 scope -- no camera/object motion), combining a current-frame
reservoir with a history reservoir via module 4's balance-heuristic combine
is unbiased **iff**, for every reservoir i, `p_hat_i,eval(x)=0` whenever
`p_hat_i,gen(x)=0` (the Coverage Lemma). For a genuine scene change
(disocclusion) this collapses to one cheap, exact, threshold-free runtime
check: `wsum_gen == 0` on the history reservoir must zero its effective
confidence in the combine **denominator**, not just skip its own candidate
-- excluding it from `combine_reservoirs` entirely (M=0) does both at once,
since `combine_reservoirs` already drops any reservoir with `M == 0`.

This module adds only that gate and the M-clamp-on-input plumbing (T23/T24,
A10) on top of module 4 -- current and history share the same pixel/vertex
domain under static geometry, so the shift maps between them are always
identity, and `combine_reservoirs`'s own `target_pdf_fn`/`contribution_weight`
split (eval-time target vs. each reservoir's own stored `p_hat_gen`) already
implements the gen/eval distinction the Coverage Lemma needs -- no new
combine arithmetic, just correct wiring.

**Flagged gap, not solved here (restir_running_notes.md §14 item 1):** no
volumetric-temporal analog of the Coverage Lemma exists. The addendum
found the volumetric case needs a genuine *practical* variance cutoff (the
transmittance-based gen density is never exactly zero at finite optical
depth, so the exact `wsum_gen==0` check never fires, unlike this surface
case) -- out of scope for this module.

**Locked v1 scope:** static geometry only (rigid-motion / non-static
geometry is Phase 2 backlog). `HistoryBuffer.reproject` is therefore always
identity -- never call it expecting per-pixel motion compensation.

Depends on: ris_reservoir (module 1), mis_combine (module 4).
"""

from ris_reservoir import Reservoir
from mis_combine import combine_reservoirs


def _identity_shift(y):
    return y, 1.0


class HistoryBuffer:
    """Per-pixel temporal reservoir history (A9/A10), keyed by pixel index.

    Locked v1 scope is static geometry only -- `reproject` is always
    identity (A10's reprojection-Jacobian-≡1 precondition holds
    unconditionally here, so pixel i's history always stays pixel i).
    """

    def __init__(self):
        self._by_pixel: dict[int, Reservoir] = {}

    def get(self, pixel_index: int) -> Reservoir:
        """History reservoir for `pixel_index`, or a fresh empty one if this
        pixel has no recorded history yet (first frame / newly exposed pixel
        -- `Reservoir()` has `M=0`, which `temporal_combine` treats exactly
        like a Coverage-Lemma-gated-out reservoir: no contribution, no bias).
        """
        return self._by_pixel.get(pixel_index, Reservoir())

    def store(self, pixel_index: int, reservoir: Reservoir) -> None:
        """Record `reservoir` as this pixel's history for the next frame."""
        self._by_pixel[pixel_index] = reservoir

    def reproject(self, motion_vectors=None) -> "HistoryBuffer":
        """Reproject history into the current frame's pixel grid.

        Always identity in the locked v1 static-geometry scope. Accepts
        `motion_vectors` only to keep this call site ready for the Phase 2
        rigid-motion backlog (restir_running_notes.md §12, out of scope for
        this paper) -- the argument is never read.
        """
        return self


def temporal_combine(current_reservoir, history_reservoir, target_pdf_fn, rng, m_cap=None, wsum_gen_gate=0.0):
    """Combine `current_reservoir` and `history_reservoir` for one pixel (A9/A10).

    `target_pdf_fn` is the single shared eval-time target for this pixel
    this frame (both reservoirs live in the same domain under static
    geometry, so no distinct per-reservoir target or shift map is needed --
    `combine_reservoirs` is called with identity shifts and the same target
    for both).

    Coverage Lemma gate: if `history_reservoir.wsum <= wsum_gen_gate`
    (default exactly 0 -- the derived, threshold-free surface-case rule,
    A9/A10), the history reservoir is excluded from the combine entirely
    (both its own candidate and its confidence in the MIS denominator),
    rather than contributing a phantom low-confidence term -- same
    drop-not-zero-fill discipline as `mis_combine`'s nonexistent-shift case.

    M-clamping (T23/T24): `m_cap` clamps the input `M` of every source
    reservoir before merging (via `combine_reservoirs`'s own `m_cap`) --
    clamping the combined output's `M` afterward bounds nothing, since
    `M·p_hat_gen(y)·W ≡ wsum` regardless of which `M` defined `W`
    (see `Reservoir.merge`'s docstring).
    """
    history_covered = history_reservoir.wsum > wsum_gen_gate
    effective_history = history_reservoir if history_covered else Reservoir()

    target_pdf_fns = [target_pdf_fn, target_pdf_fn]
    identity_row = [_identity_shift, _identity_shift]
    shift_fns = [identity_row, identity_row]

    return combine_reservoirs(
        [current_reservoir, effective_history],
        target_pdf_fns,
        shift_fns,
        dest_index=0,
        rng=rng,
        m_cap=m_cap,
    )
