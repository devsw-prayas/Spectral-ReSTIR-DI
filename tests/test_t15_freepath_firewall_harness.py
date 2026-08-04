"""Point-probe for T15 (session_log_restir_4 V-E-1: free-path firewall
harness -- is a real, lambda_s-coupled, stochastic free-path sampler
unbiased on its own?).

`tests/test_freepath_sampler.py` already checks `sample_free_path`'s
empirical hit/miss density against `free_path_pdf`/`miss_probability`
directly -- this probe goes one level up (the actual V-E-1 content): does
IMPORTANCE-SAMPLING a downstream, physically-meaningful target through the
free-path process recover an INDEPENDENT quadrature ground truth of that
target's z-integral? This is the harness A7's firewall corollary depends on
(freepath_sampler.py module docstring): before trusting cross-pixel reuse
under real random z (T16), the sampler that PRODUCES those z's must be
shown unbiased when used as an importance-sampling generation process, not
just distributionally correct in isolation.

Model: position-dependent concentration field `c(z) = C0 + C1*sigmoid(K*
(z-Z0))` (reused from T13/T14's logistic construction) drives BOTH the
free-path extinction `sigma_t(z) = c(z)` and a smooth downstream target
`TARGET(z) = c(z) * exp(-2*(z-0.5)**2)` (an arbitrary Gaussian-windowed
stand-in -- the harness's job is to check the sampler mechanics against a
target it has no special relationship to, not to re-derive a physical
model; T16 is where a real ReSTIR-shaped target gets exercised).

Native estimator (V-E-1's own form): sample z via `sample_free_path`; for a
real collision, contribution = TARGET(z)/free_path_pdf(z) (the sampler's own
density, by construction the correct denominator for the hit sub-density);
for a miss, contribution = 0. Ground truth = integral_0^1 TARGET(z) dz via
independent quadrature (separate code path from the sampler's own
optical-depth grid, per V-E-1's own "independent quadrature ground truth"
discipline).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from freepath_sampler import sample_free_path, free_path_pdf, miss_probability

torch.set_default_dtype(torch.float64)

C0, C1, K, Z0 = 0.3, 3.0, 15.0, 0.5  # logistic concentration, same family as T13


def concentration(z: float) -> float:
    return C0 + C1 / (1.0 + torch.exp(torch.tensor(-K * (z - Z0))).item())


def sigma_t_fn(z: float) -> float:
    return concentration(z)


def target(z: float) -> float:
    return concentration(z) * torch.exp(torch.tensor(-2.0 * (z - 0.5) ** 2)).item()


SIGMA_T_MAJORANT = C0 + C1  # sigmoid in (0,1), so concentration is strictly bounded by C0+C1


def _softplus(x: float) -> float:
    return torch.nn.functional.softplus(torch.tensor(x)).item()


def _optical_depth_from_zero(z: float) -> float:
    """Closed-form integral_0^z c(z')dz' -- same softplus antiderivative as T13,
    just integrated from 0 instead of to 1. Used as a fast analytic stand-in for
    `free_path_pdf`'s own quadrature (which reintegrates a 2000-point grid on
    EVERY call -- fine for the handful of cross-checks `test_freepath_sampler.py`
    already does, far too slow for a 200k-sample MC loop here)."""
    return C0 * z + (C1 / K) * (_softplus(K * (z - Z0)) - _softplus(-K * Z0))


def analytic_pdf(z: float) -> float:
    return sigma_t_fn(z) * torch.exp(torch.tensor(-_optical_depth_from_zero(z))).item()


def _quadrature_ground_truth(n_quad: int = 20_000) -> float:
    zs = torch.linspace(0.0, 1.0, n_quad)
    vals = torch.tensor([target(z.item()) for z in zs])
    return torch.trapz(vals, zs).item()


GT = _quadrature_ground_truth()


def test_majorant_bounds_sigma_t_everywhere():
    zs = torch.linspace(0.0, 1.0, 500)
    for z in zs:
        assert sigma_t_fn(z.item()) <= SIGMA_T_MAJORANT


def test_analytic_pdf_matches_free_path_pdf_quadrature():
    for z in (0.05, 0.25, 0.5, 0.75, 0.95):
        assert abs(analytic_pdf(z) - free_path_pdf(z, sigma_t_fn, 0.0)) < 1e-6


def test_hit_rate_matches_miss_probability():
    miss_p = miss_probability(sigma_t_fn, 0.0, 1.0)
    N = 50_000
    rng = torch.Generator().manual_seed(1500)
    hits = 0
    for _ in range(N):
        _, is_hit = sample_free_path(sigma_t_fn, SIGMA_T_MAJORANT, 0.0, 1.0, rng)
        hits += int(is_hit)
    empirical_miss_rate = 1.0 - hits / N
    assert abs(empirical_miss_rate - miss_p) < 0.01


def test_importance_sampled_estimator_matches_independent_quadrature():
    N = 200_000
    for seed in (7, 23):
        rng = torch.Generator().manual_seed(seed)
        samples = torch.empty(N)
        for t in range(N):
            z, is_hit = sample_free_path(sigma_t_fn, SIGMA_T_MAJORANT, 0.0, 1.0, rng)
            if is_hit:
                p_z = analytic_pdf(z)
                samples[t] = target(z) / p_z
            else:
                samples[t] = 0.0

        mean = samples.mean().item()
        stderr = samples.std().item() / (N ** 0.5)
        z_score = (mean - GT) / stderr
        assert abs(z_score) < 3.5


if __name__ == "__main__":
    test_majorant_bounds_sigma_t_everywhere()
    test_analytic_pdf_matches_free_path_pdf_quadrature()
    test_hit_rate_matches_miss_probability()
    test_importance_sampled_estimator_matches_independent_quadrature()
    print("all T15 tests passed")
