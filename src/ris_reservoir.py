"""RIS / weighted reservoir core — checklist module 1.

Streaming weighted reservoir sampling (WRS) and the two-reservoir combine
primitive (Bitterli et al. 2020, streaming RIS). Unblocks nearly every
T-item; every other module in this package builds on `Reservoir`.

Notation locked in restir_running_notes.md ("Notation locked" section):
    y            — reservoir's currently held candidate sample
    wsum         — running sum of resampling weights w_i = p_hat(x_i)/p_gen(x_i)
    M            — candidate / confidence count (c_i under M-clamping)
    p_hat_gen    — target p_hat evaluated at GENERATION time for `y`,
                   tracked separately from any later re-evaluation ("eval")
                   to support the Coverage Lemma's gen/eval split (A9)
    W            — unbiased RIS contribution weight,
                   W = wsum / (M * p_hat_gen(y))

Full multi-reservoir MIS (balance-heuristic weights `m_i(y)` across
reservoirs with *different* local targets, A8) lives in mis_combine.py —
deliberately not inlined here, since T5's bug was exactly this logic folded
into the wrong place. `Reservoir.merge` below is the MIS-free two-way
combine (single shared target, A1/A2 reconnection-valid case) that
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
        self.p_hat_gen = 0.0

    def update(self, x_i, w_i: float, rng: torch.Generator) -> bool:
        """Streaming RIS update with one candidate (x_i, w_i).

        w_i must already be the resampling weight p_hat(x_i) / p_gen(x_i).
        Returns True iff x_i was just accepted as the new `y`.
        """
        self.M += 1
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
        if accepted:
            self.p_hat_gen = p_hat_self_at_other_y
        return accepted


def ris_estimate(reservoir: Reservoir) -> float:
    """Unbiased RIS contribution weight for `reservoir` (see `Reservoir.contribution_weight`)."""
    return reservoir.contribution_weight()
