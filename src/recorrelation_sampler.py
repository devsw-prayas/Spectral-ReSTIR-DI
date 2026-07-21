"""Recorrelation lemma sampler — checklist module 2.

Tier-0 item 3 (restir_running_notes.md §1): for a rank-1 fluorescent kernel
`K_x(lambda, lambda') = e(lambda) * a(lambda')` (excitation-emission
separable), the conditional re-emission density is

    p(lambda | lambda') = e(lambda) a(lambda') / [a(lambda') * integral(e)]
                         = e(lambda) / integral(e)

— exactly independent of `lambda'`, not just in expectation. This is the
result A1/A2's reconnection-validity theorem cites for why fluorescent
vertices are exact (not approximate) reconnection points.

Also provides the product-CDF joint importance resampler for
`a(lambda') * L_e(lambda') * G` — promoted here from Inverse Paper 1's primal
sampler role to this paper's actual RIS target (see restir_running_notes.md
§0, "Core contribution"). That target feeds `Reservoir` (module 1) when
resampling the excitation wavelength lambda'.

Depends on: ris_reservoir (module 1), spectral_grid (Tier 0).
"""

import torch

from spectral_grid import SpectralGrid


def _sample_cdf(cdf: torch.Tensor, grid: SpectralGrid, rng: torch.Generator):
    """Inverse-CDF draw of one wavelength from a normalized CDF over `grid`."""
    u = torch.rand((), generator=rng)
    idx = torch.searchsorted(cdf, u).clamp(max=grid.N - 1)
    return grid.lam[idx], idx


def emission_cdf(emission_spectrum: torch.Tensor, grid: SpectralGrid) -> torch.Tensor:
    """Normalized CDF of e(lambda)/integral(e) over `grid` (quadrature via grid.weights)."""
    weighted = emission_spectrum * grid.weights
    cdf = torch.cumsum(weighted, dim=0)
    return cdf / cdf[-1]


def sample_recorrelated_lambda(
    lam_prime,
    emission_spectrum: torch.Tensor,
    grid: SpectralGrid,
    rng: torch.Generator,
):
    """Draw lambda ~ p(lambda | lambda') = e(lambda) / integral(e).

    `lam_prime` is accepted only for call-site symmetry with the rank-1
    kernel K_x = e(lambda) * a(lambda') — the Recorrelation Lemma proves this
    draw is exactly independent of `lambda'`, so it is provably unused here.
    """
    return _sample_cdf(emission_cdf(emission_spectrum, grid), grid, rng)[0]


def joint_target(
    a_lam_prime: torch.Tensor,
    L_e_lam_prime: torch.Tensor,
    G,
) -> torch.Tensor:
    """Joint importance resampling target a(lambda') * L_e(lambda') * G (A4).

    The RIS target p_hat(lambda') handed to `Reservoir` (module 1) when
    resampling the excitation wavelength lambda'.
    """
    return a_lam_prime * L_e_lam_prime * G


def joint_target_cdf(
    a_lambda_prime: torch.Tensor,
    L_e_lambda_prime: torch.Tensor,
    grid: SpectralGrid,
) -> torch.Tensor:
    """Product-CDF over lambda' for a(lambda')*L_e(lambda') (G is per-vertex constant, omitted)."""
    weighted = a_lambda_prime * L_e_lambda_prime * grid.weights
    cdf = torch.cumsum(weighted, dim=0)
    return cdf / cdf[-1]


def sample_joint_lambda_prime(
    a_lambda_prime: torch.Tensor,
    L_e_lambda_prime: torch.Tensor,
    grid: SpectralGrid,
    rng: torch.Generator,
):
    """Draw lambda' from the product-CDF joint importance sampler.

    Returns (lambda', index) — the pre-RIS candidate-generation step whose
    output feeds `Reservoir.update` with target `joint_target`.
    """
    return _sample_cdf(joint_target_cdf(a_lambda_prime, L_e_lambda_prime, grid), grid, rng)
