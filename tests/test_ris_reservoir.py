"""T-tier point-probes for module 1 (ris_reservoir.py).

Discrete 3-item target p_hat=[1,2,3] under a uniform source pdf. Checked
against the closed-form normalized target and the closed-form E[W].
"""

import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ris_reservoir import Reservoir

torch.set_default_dtype(torch.float64)

P_HAT = torch.tensor([1.0, 2.0, 3.0])
P_GEN = 1.0 / 3.0
N = 200_000
TOL = 0.01


def test_single_reservoir_matches_target_distribution():
    rng = torch.Generator().manual_seed(0)
    counts = torch.zeros(3)
    for _ in range(N):
        r = Reservoir()
        for i in range(3):
            r.update(i, P_HAT[i].item() / P_GEN, rng)
        counts[r.y] += 1

    empirical = counts / N
    expected = P_HAT / P_HAT.sum()
    assert torch.allclose(empirical, expected, atol=TOL)


def test_contribution_weight_is_unbiased():
    rng = torch.Generator().manual_seed(1)
    weights = torch.empty(N)
    for k in range(N):
        r = Reservoir()
        for i in range(3):
            r.update(i, P_HAT[i].item() / P_GEN, rng)
        r.set_p_hat_gen(P_HAT[r.y].item())
        weights[k] = r.contribution_weight()

    # E[W] estimates the domain measure under p_gen, sum(p_hat) / 1 = 3.0
    # for this uniform-over-3-items source.
    assert abs(weights.mean().item() - 3.0) < 0.05


def test_merge_matches_full_target_distribution():
    rng = torch.Generator().manual_seed(42)
    counts = torch.zeros(3)
    for _ in range(N):
        r_a = Reservoir()
        r_a.update(0, P_HAT[0].item() / P_GEN, rng)
        r_a.set_p_hat_gen(P_HAT[r_a.y].item())

        r_b = Reservoir()
        r_b.update(1, P_HAT[1].item() / P_GEN, rng)
        r_b.update(2, P_HAT[2].item() / P_GEN, rng)
        r_b.set_p_hat_gen(P_HAT[r_b.y].item())

        r_a.merge(r_b, P_HAT[r_b.y].item(), rng)
        counts[r_a.y] += 1

    empirical = counts / N
    expected = P_HAT / P_HAT.sum()
    assert torch.allclose(empirical, expected, atol=TOL)


if __name__ == "__main__":
    test_single_reservoir_matches_target_distribution()
    test_contribution_weight_is_unbiased()
    test_merge_matches_full_target_distribution()
    print("all ris_reservoir tests passed")
