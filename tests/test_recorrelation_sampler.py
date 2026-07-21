"""T-tier point-probes for module 2 (recorrelation_sampler.py).

Checks the two claims module 2 encodes:
  1. Recorrelation lemma: p(lambda|lambda') is independent of lambda' —
     sampling with two very different lambda' values must reproduce the
     SAME histogram (restir_running_notes.md Tier-0 item 3).
  2. The joint-target product-CDF sampler reproduces a(lambda')*L_e(lambda')
     up to normalization, against a small closed-form discrete grid.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spectral_grid import make_grid
from recorrelation_sampler import (
    sample_recorrelated_lambda,
    sample_joint_lambda_prime,
    joint_target_cdf,
)

torch.set_default_dtype(torch.float64)

GRID = make_grid(lam_min=400.0, lam_max=700.0, oversampling=1.0)
N = 60_000
TOL = 0.03


def _emission_spectrum(grid):
    # arbitrary smooth bump, standing in for a real fluorophore emission profile
    mu, sigma = 550.0, 40.0
    return torch.exp(-0.5 * ((grid.lam - mu) / sigma) ** 2)


def _histogram(samples, grid, nbins=10):
    edges = torch.linspace(grid.lam.min(), grid.lam.max(), nbins + 1)
    counts = torch.histogram(samples, bins=edges)[0]
    return counts / counts.sum()


def test_recorrelation_independent_of_lambda_prime():
    e = _emission_spectrum(GRID)

    rng_a = torch.Generator().manual_seed(0)
    samples_a = torch.tensor([
        sample_recorrelated_lambda(420.0, e, GRID, rng_a).item() for _ in range(N)
    ])

    rng_b = torch.Generator().manual_seed(0)
    samples_b = torch.tensor([
        sample_recorrelated_lambda(690.0, e, GRID, rng_b).item() for _ in range(N)
    ])

    # same RNG seed + same emission spectrum + different lambda_prime must give
    # bit-identical draws, since lambda_prime is provably unused.
    assert torch.equal(samples_a, samples_b)

    hist = _histogram(samples_a, GRID)
    expected = _histogram(GRID.lam[torch.multinomial(
        e * GRID.weights, N, replacement=True
    )], GRID)
    assert torch.allclose(hist, expected, atol=TOL)


def test_joint_target_sampler_matches_product_distribution():
    a = torch.linspace(0.2, 1.0, GRID.N)
    L_e = torch.exp(-0.5 * ((GRID.lam - 600.0) / 50.0) ** 2)

    rng = torch.Generator().manual_seed(1)
    idxs = torch.tensor([
        sample_joint_lambda_prime(a, L_e, GRID, rng)[1] for _ in range(N)
    ])

    empirical = torch.bincount(idxs, minlength=GRID.N).double()
    empirical /= empirical.sum()

    target_weighted = a * L_e * GRID.weights
    expected = target_weighted / target_weighted.sum()

    # coarse-bin comparison (per-index empirical counts are noisy at N samples
    # over GRID.N bins; bin down to something estimable)
    nbins = 12
    bin_edges = torch.linspace(0, GRID.N, nbins + 1).long()
    emp_binned = torch.tensor([
        empirical[bin_edges[i]:bin_edges[i + 1]].sum() for i in range(nbins)
    ])
    exp_binned = torch.tensor([
        expected[bin_edges[i]:bin_edges[i + 1]].sum() for i in range(nbins)
    ])
    assert torch.allclose(emp_binned, exp_binned, atol=TOL)


if __name__ == "__main__":
    test_recorrelation_independent_of_lambda_prime()
    test_joint_target_sampler_matches_product_distribution()
    print("all recorrelation_sampler tests passed")
