import math
import torch
from dataclasses import dataclass


@dataclass
class SpectralGrid:
    """ν̃-uniform spectral grid over [lam_min, lam_max] nm.

    nu_tilde : (N,) wavenumber [nm⁻¹], descending (1/lam_min → 1/lam_max)
    lam      : (N,) wavelength [nm],    ascending  (lam_min  → lam_max)
    weights  : (N,) trapezoid quadrature weights with λ²Δν̃ Jacobian
    N        : number of grid points (derived from Nyquist)
    d_nu     : Δν̃ spacing [nm⁻¹]
    """
    nu_tilde : torch.Tensor
    lam      : torch.Tensor
    weights  : torch.Tensor
    N        : int
    d_nu     : float


def make_grid(
    lam_min:     float = 360.0,
    lam_max:     float = 830.0,
    A:           float = 1.5,
    B:           float = 5000.0,
    d_max:       float = 3000.0,
    oversampling: float = 4.0,
    device=None,
) -> SpectralGrid:
    """Build the ν̃-uniform spectral grid.

    Parameters
    ----------
    lam_min, lam_max : wavelength range [nm].  Endpoints are parameters, not constants.
    A, B             : Cauchy coefficients  n(λ) = A + B/λ²  (λ in nm, B in nm²).
    d_max            : maximum film thickness [nm] — sets the Nyquist requirement.
    oversampling     : safety factor on top of the Nyquist minimum (≥ 1).

    Derivation
    ----------
    Fabry-Airy phase in ν̃-space: φ(ν̃) = 4π·n(ν̃)·d·cosθ·ν̃, n(ν̃) = A + B·ν̃².
    Instantaneous frequency: f(ν̃) = (1/2π)·dφ/dν̃ = 2d·cosθ·(A + 3B·ν̃²).
    Worst case at ν̃_max (blue edge), cosθ = 1 (normal incidence):

        N = ceil( oversampling × 2 × f_max × Δν̃_total )
          = ceil( oversampling × 4 · d_max · (A + 3B·ν̃_max²) · Δν̃_total )
    """
    nu_min = 1.0 / lam_max
    nu_max = 1.0 / lam_min
    d_nu_total = nu_max - nu_min

    f_max = 2.0 * d_max * (A + 3.0 * B * nu_max ** 2)
    N = max(math.ceil(oversampling * 2.0 * f_max * d_nu_total), 2)

    # ν̃ descending → lam = 1/ν̃ ascending (360 → 830 nm)
    nu = torch.linspace(nu_max, nu_min, N, dtype=torch.float64, device=device)
    lam = 1.0 / nu
    d_nu = d_nu_total / (N - 1)

    # trapezoid weights: w_k = λ_k² · Δν̃, halved at endpoints
    weights = lam ** 2 * d_nu
    weights[0]  *= 0.5
    weights[-1] *= 0.5

    return SpectralGrid(nu_tilde=nu, lam=lam, weights=weights, N=N, d_nu=d_nu)
