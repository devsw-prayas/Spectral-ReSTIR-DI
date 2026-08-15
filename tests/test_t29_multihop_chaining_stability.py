"""Point-probe for T29 (multi-hop chaining stability).

Question: does the composition lemma (T26-T28: `J_total = J_reproj`,
spectral part via detached fresh resampling) survive being CHAINED across
many sequential frames -- both floating-point compounding (many
multiplications accumulating rounding error) and statistical drift (does
bias creep in as the chain grows)?

Setup: a family of 1D reprojection maps `phi_t(u)` indexed by hop count
`t`, deformation magnitude growing LINEARLY with `t` up to `t=T_MAX=500`
(`alpha(t) = t/T_MAX`), `phi_0 = identity` (`J_0 == 1` everywhere) per the
historical session's own convention. `phi_t(u) = u + alpha(t)*0.3*sin(pi*u)`
is the same family shape T26/T29's own asymmetric-weight-field lesson
would recommend checking, but here the deformation shape itself is
symmetric -- deliberately fine for THIS test, since T29's question is
purely about compounding-error/drift with hop count, not about a
cancellation trap (there is no "broken estimator" being tested here, only
whether the CORRECT chained construction stays correct).

**Part A -- floating-point robustness**: for `T` in {10, 50, 100, 200,
500}, compare a SEQUENTIALLY-CHAINED Jacobian-ratio product
(`prod_{k=1}^{T} J_k(u)/J_{k-1}(u)`, accumulating one division and one
multiplication per hop -- simulating a real per-frame history chain, where
each frame only ever has access to the PREVIOUS frame's Jacobian, not the
original frame-0 reference) against the mathematically equivalent DIRECT
single-shot ratio `J_T(u)/J_0(u)` computed once. These telescope to the
identical value in exact arithmetic; the test is whether T sequential
floating-point multiplications introduce meaningful compounding error.
Historical result: max relative error grows slowly (~sqrt(T)-like) but
stays at the machine-precision floor (`~1e-15`) through 500 hops --
reconfirmed here with fresh parameters.

**Part B -- statistical drift**: does an actual self-normalized-free MC
estimator, reweighted by the CHAINED (not direct) Jacobian and drawing a
fresh spectral sample at each hop's destination, stay unbiased for the
SAME closed-form truth (`TRUTH=550`, same asymmetric species-weight field
and closed-form derivation as T26) as the chain grows from `T=1` to
`T=500`? If chaining introduced any systematic per-hop bias, it should
show up as bias/SEM drifting away from zero with growing `T` -- checked
directly, not assumed.

**Performance note (matches the historical session's own caught bug):**
the historical first attempt recomputed the chain in a way that redundantly
re-evaluated work without reusing intermediate results, adding ~19s of
dead weight at `T=500`. This implementation loops only over hop count `T`
(<=500 iterations), vectorized over all `N` sample points per hop via
`torch` -- avoid re-looping over `N` inside the hop loop, or this file will
hit the same class of slowdown at `T=500`.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

torch.set_default_dtype(torch.float64)

T_MAX = 500
MU1, MU2, SIGMA = 480.0, 620.0, 15.0
TRUTH = 550.0  # same closed form as T26: MU2 - (MU2-MU1)*integral_0^1 w1(x)dx


def phi_hop(u, t):
    alpha = t / T_MAX
    return u + alpha * 0.3 * torch.sin(torch.pi * u)


def phi_hop_prime(u, t):
    alpha = t / T_MAX
    return 1.0 + alpha * 0.3 * torch.pi * torch.cos(torch.pi * u)


def chained_jacobian(u, T):
    """Sequential product of per-hop ratios J_k(u)/J_{k-1}(u), k=1..T --
    simulates a real history chain that only ever has the previous frame's
    Jacobian on hand, not a cached frame-0 reference."""
    chain = torch.ones_like(u)
    j_prev = phi_hop_prime(u, 0)
    for k in range(1, T + 1):
        j_k = phi_hop_prime(u, k)
        chain = chain * (j_k / j_prev)
        j_prev = j_k
    return chain


def w1(x):
    return 0.1 + 0.8 * x


def mean_lambda(x):
    return MU2 - (MU2 - MU1) * w1(x)


def sample_mixture(x, rng):
    n = x.shape[0]
    p1 = w1(x)
    is_species1 = torch.rand(n, generator=rng) < p1
    mu = torch.where(is_species1, torch.tensor(MU1), torch.tensor(MU2))
    return mu + SIGMA * torch.randn(n, generator=rng)


def test_phi_hop_is_identity_at_t_zero():
    grid = torch.linspace(0.0, 1.0, 1001)
    assert torch.allclose(phi_hop(grid, 0), grid)
    assert torch.allclose(phi_hop_prime(grid, 0), torch.ones_like(grid))


def test_phi_hop_stays_a_valid_bijection_at_max_deformation():
    grid = torch.linspace(0.0, 1.0, 50_001)
    assert bool((phi_hop_prime(grid, T_MAX) > 0).all())
    assert abs(phi_hop(torch.tensor(0.0), T_MAX).item()) < 1e-12
    assert abs(phi_hop(torch.tensor(1.0), T_MAX).item() - 1.0) < 1e-12


def test_chained_jacobian_matches_direct_ratio_at_machine_precision():
    N = 100_000
    rng = torch.Generator().manual_seed(2900)
    u = torch.rand(N, generator=rng)
    for T in (10, 50, 100, 200, 500):
        chain = chained_jacobian(u, T)
        direct = phi_hop_prime(u, T) / phi_hop_prime(u, 0)
        rel_err = ((chain - direct).abs() / direct.abs()).max().item()
        assert rel_err < 1e-9, f"T={T}: rel_err={rel_err}"


def test_chained_estimator_shows_no_statistical_drift_with_hop_count():
    N = 100_000
    for T in (1, 10, 50, 100, 200, 500):
        rng = torch.Generator().manual_seed(2900 + T)
        u = torch.rand(N, generator=rng)
        weight = chained_jacobian(u, T)
        x_dest = phi_hop(u, T)
        lam = sample_mixture(x_dest, rng)
        samples = lam * weight

        mean = samples.mean().item()
        sem = samples.std().item() / (N ** 0.5)
        z = (mean - TRUTH) / sem
        assert abs(z) < 3.5, f"T={T}: z={z}"


if __name__ == "__main__":
    test_phi_hop_is_identity_at_t_zero()
    test_phi_hop_stays_a_valid_bijection_at_max_deformation()
    test_chained_jacobian_matches_direct_ratio_at_machine_precision()
    test_chained_estimator_shows_no_statistical_drift_with_hop_count()
    print("all T29 tests passed")
