"""RIS / weighted reservoir core — checklist module 1.

Streaming weighted reservoir sampling (WRS) and the two-reservoir combine
primitive (Bitterli et al. 2020, streaming RIS). Unblocks nearly every
T-item; every other module in this package builds on `Reservoir`.

Field notation used throughout this module:
    y            — reservoir's currently held candidate sample
    wsum         — running sum of resampling weights w_i = p_hat(x_i)/p_gen(x_i)
    M            — RAW streamed-candidate count. This is the divisor
                   `contribution_weight()` uses (`wsum/(M*p_hat_gen)`), and
                   it is always 1 on a `mis_combine.combine_reservoirs`
                   output, because that output's `wsum` is already the
                   fully-shaded `p_hat(y)*W` value, not a raw multi-candidate
                   sum that still needs dividing by a candidate count.
    confidence   — pooled/capped confidence (c_i under M-clamping), the
                   count `mis_combine.balance_heuristic_weight` reads for its
                   MIS ratio. Mirrors `M` for a raw streamed reservoir (every
                   accepted `update`/`merge` call keeps them equal); a
                   combine's output instead sets `M=1` and `confidence` to
                   the pooled, `m_cap`-clamped source count. `M` and
                   `confidence` had to be split into two fields because they
                   serve two genuinely different roles once combines chain
                   across frames -- one raw candidate count for the current
                   reservoir's own normalization, one pooled/capped count for
                   the next combine's MIS ratio -- and a single shared field
                   can't hold both correctly at once.
    p_hat_gen    — target p_hat evaluated at GENERATION time for `y`,
                   tracked separately from any later re-evaluation ("eval")
                   so a combine can weight a reservoir by what its candidate
                   looked like when it was drawn, not by how it scores under
                   a target that has since moved on
    W            — unbiased RIS contribution weight,
                   W = wsum / (M * p_hat_gen(y))

Full multi-reservoir MIS (balance-heuristic weights `m_i(y)` across
reservoirs with *different* local targets) lives in mis_combine.py —
deliberately not inlined here: a real bug once lived in exactly this logic
when it was folded into reservoir-merge code, and separating it out is what
made that bug provable and fixable. `Reservoir.merge` below is the MIS-free
two-way combine (single shared target, identity-shift case) that
mis_combine.py calls once per term with the real m_i weight in place of the
plain M it uses on its own.
"""

import torch


class Reservoir:
    """Single-sample streaming RIS reservoir."""

    def __init__(self):
        self.y = None
        self.wsum = 0.0
        self.M = 0
        self.confidence = 0
        self.p_hat_gen = 0.0

    def update(self, x_i, w_i: float, rng: torch.Generator) -> bool:
        """Streaming RIS update with one candidate (x_i, w_i).

        w_i must already be the resampling weight p_hat(x_i) / p_gen(x_i).
        Returns True iff x_i was just accepted as the new `y`.
        """
        self.M += 1
        self.confidence += 1
        self.wsum += w_i
        if self.wsum <= 0.0:
            return False
        accepted = torch.rand((), generator=rng).item() < (w_i / self.wsum)
        if accepted:
            self.y = x_i
        return accepted

    def set_p_hat_gen(self, p_hat_gen: float) -> None:
        """Record p_hat(y) at generation time (A9 gen/eval split)."""
        self.p_hat_gen = p_hat_gen

    def contribution_weight(self) -> float:
        """W = wsum / (M * p_hat_gen(y)) — unbiased RIS weight."""
        if self.M == 0 or self.p_hat_gen == 0.0:
            return 0.0
        return self.wsum / (self.M * self.p_hat_gen)

    def merge(
        self,
        other: "Reservoir",
        p_hat_self_at_other_y: float,
        rng: torch.Generator,
        m_cap: int | None = None,
    ) -> bool:
        """Fold `other` into `self` (spatial/temporal reuse primitive, MIS-free).

        `other`'s chosen sample is treated as a single candidate with
        resampling weight `p_hat_self_at_other_y * other.W * other.M`, then
        the remaining M-1 confidence is absorbed directly — standard
        streaming-RIS combine.

        M-clamping (T23/T24, A10): clamp the INPUT `other.M` via `m_cap`
        before merging, never the output — clamping wsum_combine's resulting
        M after the fact bounds nothing, since M*p_hat(y)*W ≡ wsum
        regardless of which M defined W.
        """
        other_M = other.M if m_cap is None else min(other.M, m_cap)
        w = p_hat_self_at_other_y * other.contribution_weight() * other_M
        accepted = self.update(other.y, w, rng)
        self.M += other_M - 1
        self.confidence += other_M - 1
        if accepted:
            self.p_hat_gen = p_hat_self_at_other_y
        return accepted


def ris_estimate(reservoir: Reservoir) -> float:
    """Unbiased RIS contribution weight for `reservoir` (see `Reservoir.contribution_weight`)."""
    return reservoir.contribution_weight()
