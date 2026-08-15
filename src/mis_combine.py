"""MIS balance-heuristic combine — checklist module 4.

Standalone, independently testable unit. Deliberately NOT inlined into
`Reservoir.merge` or `Reservoir` itself: a real bug (a hard support-mismatch
under a dispersive cutoff) once lived in exactly this logic, and the fix
only became provable once this was pulled out on its own. Any future edit to
reservoir combine logic must not fold this back in.

Generalized balance-heuristic MIS weight, for combining reservoirs that each
have their own local target and their own domain (reached from each other
via a shift map T_{i->j}):

    m_i(y) = c_i * p_hat_i(y) / Sum_j c_j * p_hat_j(T_{i->j}(y)) * |dT_{i->j}/dy|

At any vertex where the reconnection between domains is the identity map
(`shift_maps.is_reconnection_valid` == True, T==id, J==1) this collapses to
the ordinary balance heuristic:

    m_i(y) = c_i * p_hat_i(y) / Sum_j c_j * p_hat_j(y)

Getting this formula wrong (using the collapsed form even where the
reconnection is genuinely non-identity) silently reintroduces the same bias
the general form exists to avoid — this module's whole reason for existing
is to keep that distinction correct in one place.

Depends on: ris_reservoir (module 1), shift_maps (module 3).
"""

import torch

from ris_reservoir import Reservoir


def balance_heuristic_weight(index, y_in_domain_i, reservoirs, target_pdf_fns, shift_fns, m_cap=None):
    """Generalized balance-heuristic MIS weight m_i(y).

    `y_in_domain_i` is `y` expressed in reservoir `index`'s own domain.
    `shift_fns[index][j]` is the shift map T_{index->j}(y) -> (y_j, J), or
    `(None, 0.0)` if the reconnection doesn't exist at domain j (dispersive
    TIR, or any other support failure) — such terms drop out of the sum
    entirely: a missing reconnection is an existence failure, not a
    zero-weight candidate that should be silently zero-filled into the sum.

    Reads each reservoir's `confidence`, not `M` — `M` is reserved for
    `contribution_weight()`'s own normalization divisor and is always 1 on a
    prior combine's output (see the module-1 docstring for why those two
    fields are kept separate). `m_cap` clamps each reservoir's confidence on
    the way into the ratio, per-source, matching `Reservoir.merge`'s
    existing input-side clamp convention.
    """
    def _conf(r):
        return r.confidence if m_cap is None else min(r.confidence, m_cap)

    c_i = _conf(reservoirs[index])
    p_hat_i = target_pdf_fns[index](y_in_domain_i)
    numerator = c_i * p_hat_i

    denom = 0.0
    for j, target_pdf_fn in enumerate(target_pdf_fns):
        y_j, J = shift_fns[index][j](y_in_domain_i)
        if y_j is None:
            continue
        denom += _conf(reservoirs[j]) * target_pdf_fn(y_j) * abs(J)

    if denom == 0.0:
        return 0.0
    return numerator / denom


def combine_reservoirs(reservoirs, target_pdf_fns, shift_fns, dest_index, rng, m_cap=None):
    """MIS-weighted streaming combine of `reservoirs` into domain `dest_index`.

    For each source reservoir i, its sample y_i (domain i) is reconnected to
    the destination domain via `shift_fns[i][dest_index]`; skipped entirely
    (no stream, no confidence added) if that reconnection doesn't exist. The
    destination's own reservoir (i == dest_index, identity shift) always
    survives this filter, so the combined output always has at least that
    one term's support covered.

    Streamed resampling weight for source i's reconnected sample:
        w_i = m_i(y_i) * p_hat_dest(y_i_dest) * |J_i| * W_i
    where `m_i` is `balance_heuristic_weight` evaluated in i's own domain and
    `W_i = reservoirs[i].contribution_weight()`.

    M-clamping: clamp each source's confidence via `m_cap` on the way in,
    never the combined output's.

    `combined.wsum` is already the fully-shaded `p_hat(y)*W` value, not a raw
    multi-candidate sum — so the output's `M` (the normalization divisor
    `contribution_weight()` uses) is pinned to 1, never the pooled source
    count. That pooled count lives in `combined.confidence` instead — the
    sum of each source's own `m_cap`-clamped confidence (the clamp applies
    per-source on the way in, not to the pooled total afterward —
    `T_total = Sum_i min(c_i, m_cap)`, no second clamp on that sum) — the
    field the *next* combine round's `balance_heuristic_weight` ratio reads.
    Storing the pooled count in `.M` instead corrupts `contribution_weight()`
    the instant this output becomes a source in a later combine call, which
    is exactly why the two fields are kept separate (see the module-1
    docstring).
    """
    combined = Reservoir()
    total_confidence = 0.0
    for i, r_i in enumerate(reservoirs):
        if r_i.y is None or r_i.confidence == 0:
            continue

        m_i = balance_heuristic_weight(i, r_i.y, reservoirs, target_pdf_fns, shift_fns, m_cap)
        y_dest, J_i = shift_fns[i][dest_index](r_i.y)
        if y_dest is None:
            continue

        c_i = r_i.confidence if m_cap is None else min(r_i.confidence, m_cap)
        p_hat_dest = target_pdf_fns[dest_index](y_dest)
        w_i = m_i * p_hat_dest * abs(J_i) * r_i.contribution_weight()

        accepted = combined.update(y_dest, w_i, rng)
        total_confidence += c_i
        if accepted:
            combined.set_p_hat_gen(p_hat_dest)

    combined.M = 1
    combined.confidence = total_confidence
    return combined
