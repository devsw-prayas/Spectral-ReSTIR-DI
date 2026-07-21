"""T-tier point-probes for module 4 (mis_combine.py).

Covers the A8 claim this module exists to get right: the generalized
balance heuristic collapses to the ordinary balance heuristic wherever
reconnection is valid (T==id, J==1), and a 3-way `combine_reservoirs` over
a shared target reduces to the same pooled distribution as module 1's
pairwise `Reservoir.merge` test. Also checks that a non-existent shift
(TIR / support-mismatch surrogate) drops a source out entirely rather than
contributing a phantom zero-weight term.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ris_reservoir import Reservoir
from mis_combine import balance_heuristic_weight, combine_reservoirs

torch.set_default_dtype(torch.float64)


def _identity(y):
    return y, 1.0


def _none_shift(y):
    return None, 0.0


def test_balance_heuristic_collapses_to_ordinary_when_reconnection_valid():
    # 3 reservoirs, different local targets, but reconnection-valid (identity
    # shift) everywhere -- A8's collapse case.
    P = [
        {0: 1.0, 1: 2.0, 2: 3.0},
        {0: 3.0, 1: 2.0, 2: 1.0},
        {0: 2.0, 1: 2.0, 2: 2.0},
    ]
    target_pdf_fns = [lambda y, p=p: p[y] for p in P]
    reservoirs = []
    for c in (2, 5, 3):
        r = Reservoir()
        r.M = c
        reservoirs.append(r)

    shift_fns = [[_identity, _identity, _identity] for _ in range(3)]

    y = 1
    for i in range(3):
        got = balance_heuristic_weight(i, y, reservoirs, target_pdf_fns, shift_fns)
        c = [2, 5, 3]
        expected = c[i] * P[i][y] / sum(c[j] * P[j][y] for j in range(3))
        assert abs(got - expected) < 1e-12


def test_combine_matches_pooled_target_distribution():
    # Same discrete 3-item target as module 1's merge test, but split across
    # 3 single-candidate reservoirs and combined via the general n-way MIS
    # machinery instead of pairwise Reservoir.merge.
    p_hat = {0: 1.0, 1: 2.0, 2: 3.0}
    p_gen = 1.0 / 3.0
    target_pdf_fns = [lambda y: p_hat[y]] * 3
    identity_row = [_identity, _identity, _identity]
    shift_fns = [identity_row, identity_row, identity_row]

    N = 200_000
    rng = torch.Generator().manual_seed(7)
    counts = torch.zeros(3)
    for _ in range(N):
        reservoirs = []
        for val in (0, 1, 2):
            r = Reservoir()
            r.update(val, p_hat[val] / p_gen, rng)
            r.set_p_hat_gen(p_hat[val])
            reservoirs.append(r)

        combined = combine_reservoirs(reservoirs, target_pdf_fns, shift_fns, dest_index=0, rng=rng)
        counts[combined.y] += 1

    empirical = counts / N
    expected = torch.tensor([p_hat[0], p_hat[1], p_hat[2]])
    expected = expected / expected.sum()
    assert torch.allclose(empirical, expected, atol=0.01)


def test_nonexistent_shift_drops_source_entirely():
    # reservoir A: reconnection-valid, single candidate y=7
    r_a = Reservoir()
    r_a.update(7, 5.0, torch.Generator().manual_seed(0))
    r_a.set_p_hat_gen(5.0)

    # reservoir B: its shift into the destination never exists (TIR surrogate)
    r_b = Reservoir()
    r_b.update(3, 9.0, torch.Generator().manual_seed(1))
    r_b.set_p_hat_gen(9.0)

    target_pdf_fns = [lambda y: 5.0, lambda y: 9.0]
    shift_fns = [
        [_identity, _identity],
        [_none_shift, _identity],
    ]

    rng = torch.Generator().manual_seed(2)
    combined = combine_reservoirs([r_a, r_b], target_pdf_fns, shift_fns, dest_index=0, rng=rng)

    # only r_a could ever stream into dest_index=0 -- deterministic outcome
    assert combined.y == 7
    assert combined.M == r_a.M  # r_b contributed zero confidence, not just zero weight


if __name__ == "__main__":
    test_balance_heuristic_collapses_to_ordinary_when_reconnection_valid()
    test_combine_matches_pooled_target_distribution()
    test_nonexistent_shift_drops_source_entirely()
    print("all mis_combine tests passed")
