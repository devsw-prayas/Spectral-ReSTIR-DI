"""Symbolic (SymPy) proof of T32's merge-order-invariance claim
(`addendum_session11_13_test_plan_extension.md`, "T32 -- compound
spatiotemporal, merge-order invariance"): under T32's exact scope (every
source shares one eval-time target, every pairwise shift is identity, no
`m_cap`), `combine_reservoirs`' output `.wsum`/`.confidence` are provably
independent of how the sources are grouped into intermediate combine calls.

`test_t32_merge_order_invariance.py` confirms this only at concrete
floating-point values, across a handful of random seeds (`~1e-9` relative
tolerance, matching the historical session's own `~1e-15` finding). This file
proves the stronger, seed-independent claim: a 2-level-nesting symbolic proof
(`project_infra_checkpoint.md`'s own suggested next step for this item),
generalized to an inductive argument for ARBITRARY nesting depth.

**Building block 1 (single combine-call formula, this repo's actual code):**
combined via `mis_combine.combine_reservoirs` under this scope, using T22's
own proven collapse `m_i = c_i / Sum_j c_j` (see
`test_a_balance_heuristic_shared_target_collapse.py`) substituted into the
`w_i = m_i * p_hat_dest(y_dest) * |J_i| * W_i` formula, plus the identity
`p_hat_dest(y_dest) * W_i == wsum_i / M_i` that holds whenever a reservoir's
`p_hat_gen` equals the shared target evaluated at its own `y` (true by
construction for every reservoir in this family -- raw reservoirs set it at
acceptance time via the same shared target; combine outputs set it via
`set_p_hat_gen(p_hat_dest)`, same shared target again) -- gives, for sources
`{r_i}` with confidence `c_i` and `v_i := wsum_i / M_i`:

    wsum_out      = Sum_i(c_i * v_i) / Sum_j(c_j)
    confidence_out = Sum_j(c_j)
    M_out          = 1                              (pinned, per module docstring)

**Building block 2 (the invariant that makes nesting transparent):** because
`M_out` is always pinned to 1 on a combine's output, `v_out := wsum_out/M_out`
equals `wsum_out` itself for any non-leaf source -- so a combine's own output
can be fed back into building block 1 as just another `(v_i, c_i)` pair with
no special-casing. For RAW leaf reservoirs, `c_i == M_i` (confidence mirrors
M until the first combine), so `c_i * v_i == c_i * (wsum_i/M_i) == wsum_i`
exactly -- the `M_i` cancels.

**The induction (why merge order cannot matter, arbitrary tree depth):**
claim -- for ANY reservoir `r` produced by ANY tree of nested
`combine_reservoirs` calls over a fixed multiset of raw leaves L(r), the pair
`(v(r), confidence(r))` equals `(Sum_{leaf in L(r)} wsum_leaf / Sum_{leaf in
L(r)} M_leaf, Sum_{leaf in L(r)} M_leaf)` -- depending only on the SET of raw
leaves beneath `r`, never on the shape of the tree above them.
  - Base case (r is a raw leaf): trivially true, L(r) = {r}.
  - Inductive step: by hypothesis, each child `r_i` already satisfies the
    claim, i.e. `c_i = Sum_{leaf in L(r_i)}(M_leaf)` and
    `v_i = Sum_{leaf in L(r_i)}(wsum_leaf) / c_i`. Then
    `c_i * v_i = Sum_{leaf in L(r_i)}(wsum_leaf)` exactly (the `c_i`
    cancels), so `Sum_i(c_i * v_i) = Sum_{leaf in L(r)}(wsum_leaf)` and
    `Sum_i(c_i) = Sum_{leaf in L(r)}(M_leaf)` -- building block 1 gives the
    parent the same claim, closing the induction.

Two different tree shapes over the same leaf set (e.g. temporal-then-spatial
vs. spatial-then-temporal vs. flat) therefore always reach the identical
`(v(root), confidence(root))` at the root -- this is a structural fact about
the recursion, not a numerical coincidence needing per-seed verification.

This file mechanizes the inductive step's algebra (building blocks 1-2) via
SymPy for the specific 2-level nesting T32 itself tests (a 2x2 grid, all
three of T32's own comparisons: flat, temporal-first, spatial-first), with
fully general (symbolic, not numeric) per-cell `wsum`/`M` values -- the
concrete instance the general induction argument above specializes to.
"""

import sympy as sp


def _raw_combine(items):
    """Building block 1, raw-leaf case (c_i == M_i, cancels): returns
    (wsum_out, confidence_out) for a group of raw (wsum_i, M_i) leaves."""
    num = sum(wsum_i for wsum_i, m_i in items)
    den = sum(m_i for wsum_i, m_i in items)
    return num / den, den


def _combined_combine(items):
    """Building block 1, general case: returns (wsum_out, confidence_out)
    for a group of (v_i, c_i) sources (raw or already-combined, per the
    v := wsum/M identity in the module docstring -- M_out is always 1, so
    wsum_out doubles as v_out for the next level up)."""
    num = sum(c_i * v_i for v_i, c_i in items)
    den = sum(c_i for v_i, c_i in items)
    return num / den, den


def _make_grid_symbols():
    w, m = {}, {}
    for t in (0, 1):
        for n in (0, 1):
            w[t, n] = sp.symbols(f"w_{t}{n}", positive=True)
            m[t, n] = sp.symbols(f"M_{t}{n}", positive=True)
    return w, m


def test_flat_equals_temporal_then_spatial():
    w, m = _make_grid_symbols()

    flat_wsum, flat_conf = _raw_combine([(w[t, n], m[t, n]) for t in (0, 1) for n in (0, 1)])

    columns = [_raw_combine([(w[t, n], m[t, n]) for t in (0, 1)]) for n in (0, 1)]
    temporal_then_spatial_wsum, temporal_then_spatial_conf = _combined_combine(columns)

    assert sp.simplify(flat_wsum - temporal_then_spatial_wsum) == sp.Integer(0)
    assert sp.simplify(flat_conf - temporal_then_spatial_conf) == sp.Integer(0)


def test_flat_equals_spatial_then_temporal():
    w, m = _make_grid_symbols()

    flat_wsum, flat_conf = _raw_combine([(w[t, n], m[t, n]) for t in (0, 1) for n in (0, 1)])

    rows = [_raw_combine([(w[t, n], m[t, n]) for n in (0, 1)]) for t in (0, 1)]
    spatial_then_temporal_wsum, spatial_then_temporal_conf = _combined_combine(rows)

    assert sp.simplify(flat_wsum - spatial_then_temporal_wsum) == sp.Integer(0)
    assert sp.simplify(flat_conf - spatial_then_temporal_conf) == sp.Integer(0)


def test_all_three_reduce_to_total_raw_wsum_over_total_raw_m():
    # Confirms the closed form itself (not just mutual equality): the common
    # value all three orderings reach is literally
    # Sum(all raw wsum)/Sum(all raw M) -- the exact quantity
    # `project_infra_checkpoint.md`'s T32 entry states by hand.
    w, m = _make_grid_symbols()
    flat_wsum, flat_conf = _raw_combine([(w[t, n], m[t, n]) for t in (0, 1) for n in (0, 1)])

    total_wsum = sum(w.values())
    total_m = sum(m.values())

    assert sp.simplify(flat_wsum - total_wsum / total_m) == sp.Integer(0)
    assert sp.simplify(flat_conf - total_m) == sp.Integer(0)


def test_three_level_nesting_also_invariant():
    # Extends beyond T32's own 2-level scope: a 2x2x2 cube (e.g. T frames x
    # M neighbors x K a third pooling axis), combined in two different
    # 3-level tree shapes, to confirm the induction argument's
    # depth-independence claim isn't an artifact of exactly 2 levels.
    w, m = {}, {}
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                w[a, b, c] = sp.symbols(f"w_{a}{b}{c}", positive=True)
                m[a, b, c] = sp.symbols(f"M_{a}{b}{c}", positive=True)

    flat_wsum, flat_conf = _raw_combine(
        [(w[a, b, c], m[a, b, c]) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    )

    # Shape 1: combine over c first (leaf groups), then over b, then over a.
    level1 = {
        (a, b): _raw_combine([(w[a, b, c], m[a, b, c]) for c in (0, 1)])
        for a in (0, 1)
        for b in (0, 1)
    }
    level2 = {a: _combined_combine([level1[a, b] for b in (0, 1)]) for a in (0, 1)}
    shape1_wsum, shape1_conf = _combined_combine([level2[a] for a in (0, 1)])

    # Shape 2: combine over a first, then over c, then over b -- a genuinely
    # different tree, same leaf multiset.
    level1b = {
        (b, c): _raw_combine([(w[a, b, c], m[a, b, c]) for a in (0, 1)])
        for b in (0, 1)
        for c in (0, 1)
    }
    level2b = {b: _combined_combine([level1b[b, c] for c in (0, 1)]) for b in (0, 1)}
    shape2_wsum, shape2_conf = _combined_combine([level2b[b] for b in (0, 1)])

    assert sp.simplify(flat_wsum - shape1_wsum) == sp.Integer(0)
    assert sp.simplify(flat_wsum - shape2_wsum) == sp.Integer(0)
    assert sp.simplify(flat_conf - shape1_conf) == sp.Integer(0)
    assert sp.simplify(flat_conf - shape2_conf) == sp.Integer(0)


if __name__ == "__main__":
    test_flat_equals_temporal_then_spatial()
    test_flat_equals_spatial_then_temporal()
    test_all_three_reduce_to_total_raw_wsum_over_total_raw_m()
    test_three_level_nesting_also_invariant()
    print("all A-merge-order-associativity tests passed")
