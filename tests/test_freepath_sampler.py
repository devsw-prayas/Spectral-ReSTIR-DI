"""T-tier point-probes for module 5 (freepath_sampler.py).

Harness validation: is the delta-tracking sampler itself unbiased, checked
against an independent closed-form / quadrature ground truth? Two cases:
homogeneous (majorant exact, reduces to plain exponential sampling) and
heterogeneous (majorant loose, exercises null-collision rejection).
"""

import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from freepath_sampler import sample_free_path, free_path_pdf, miss_probability

torch.set_default_dtype(torch.float64)

N = 200_000
TOL = 0.02


def test_homogeneous_miss_probability_matches_closed_form():
    sigma_t = 2.0
    rng = torch.Generator().manual_seed(0)
    misses = 0
    for _ in range(N):
        _, hit = sample_free_path(lambda z: sigma_t, sigma_t, 0.0, 1.0, rng)
        if not hit:
            misses += 1

    empirical = misses / N
    expected = torch.exp(torch.tensor(-sigma_t)).item()  # exp(-2) ~ 0.1353
    assert abs(empirical - expected) < TOL


def test_homogeneous_hit_distribution_matches_exponential_pdf():
    sigma_t = 2.0
    rng = torch.Generator().manual_seed(1)
    hits = []
    for _ in range(N):
        z, hit = sample_free_path(lambda z: sigma_t, sigma_t, 0.0, 1.0, rng)
        if hit:
            hits.append(z)
    hits = torch.tensor(hits)

    edges = torch.linspace(0.0, 1.0, 11)
    counts = torch.histogram(hits, bins=edges)[0]
    empirical_density = counts / (len(hits) * (edges[1] - edges[0]))

    midpoints = 0.5 * (edges[:-1] + edges[1:])
    # unnormalized exponential hit density (matches sample_free_path's own
    # convention: relative shape sigma_t*exp(-sigma_t*z), renormalized over
    # hits only)
    raw = sigma_t * torch.exp(-sigma_t * midpoints)
    expected_density = raw / raw.sum() / (edges[1] - edges[0])

    assert torch.allclose(empirical_density, expected_density, atol=TOL * 5)


def test_heterogeneous_matches_quadrature_ground_truth():
    # sigma_t(z) = 1 + z, majorant = 2.0 (max over [0,1]) -- exercises real
    # null-collision rejection, not just the homogeneous fast path.
    sigma_t_fn = lambda z: 1.0 + z
    majorant = 2.0

    rng = torch.Generator().manual_seed(2)
    misses = 0
    hits = []
    for _ in range(N):
        z, hit = sample_free_path(sigma_t_fn, majorant, 0.0, 1.0, rng)
        if hit:
            hits.append(z)
        else:
            misses += 1
    hits = torch.tensor(hits)

    empirical_miss = misses / N
    expected_miss = miss_probability(sigma_t_fn, 0.0, 1.0)
    assert abs(empirical_miss - expected_miss) < TOL

    # spot-check hit density at a few z against the quadrature pdf, coarse
    # binned (same discipline as the recorrelation-sampler tests)
    edges = torch.linspace(0.0, 1.0, 9)
    counts = torch.histogram(hits, bins=edges)[0]
    empirical_density = counts / (len(hits) * (edges[1] - edges[0])) * (1.0 - empirical_miss)

    midpoints = 0.5 * (edges[:-1] + edges[1:])
    expected_density = torch.tensor([free_path_pdf(m.item(), sigma_t_fn, 0.0) for m in midpoints])

    assert torch.allclose(empirical_density, expected_density, atol=TOL * 5)


if __name__ == "__main__":
    test_homogeneous_miss_probability_matches_closed_form()
    test_homogeneous_hit_distribution_matches_exponential_pdf()
    test_heterogeneous_matches_quadrature_ground_truth()
    print("all freepath_sampler tests passed")
