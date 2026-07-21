import torch


def n_cauchy(lam: torch.Tensor, A: torch.Tensor | float, B: torch.Tensor | float) -> torch.Tensor:
    """Cauchy IOR  n(λ) = A + B / λ²,  λ in nm."""
    return A + B / (lam ** 2)


def dn_dlam(lam: torch.Tensor, B: torch.Tensor | float) -> torch.Tensor:
    """dn/dλ = -2B / λ³."""
    return -2.0 * B / (lam ** 3)


def critical_angle(n_i: torch.Tensor, n_t: torch.Tensor) -> torch.Tensor:
    """Critical angle θ_c = arcsin(n_t / n_i) for each wavelength.

    Returns NaN where n_t >= n_i (no TIR).
    """
    ratio = n_t / n_i
    # arcsin only defined on [-1, 1]; clamp to signal TIR via NaN
    safe = torch.where(ratio < 1.0, ratio, torch.full_like(ratio, float("nan")))
    return torch.asin(safe)


def cos_theta_t(cos_theta_i: torch.Tensor, n_i: torch.Tensor | float, n_t: torch.Tensor | float) -> torch.Tensor:
    """cosθ_t from Snell's law: cos θ_t = sqrt(1 - (n_i/n_t)² (1 - cos²θ_i)).

    Returns real values in (0, 1] for propagating rays, 0.0 at TIR onset,
    and imaginary-magnitude values (returned as 0.0) for TIR wavelengths —
    caller checks via is_tir().
    """
    sin2_i = 1.0 - cos_theta_i ** 2
    sin2_t = (n_i / n_t) ** 2 * sin2_i
    under = 1.0 - sin2_t
    # clamp to 0: cosθ_t = 0 at TIR onset; negative values mean TIR
    return torch.sqrt(under.clamp(min=0.0))


def is_tir(cos_theta_i: torch.Tensor, n_i: torch.Tensor | float, n_t: torch.Tensor | float) -> torch.Tensor:
    """Boolean mask: True where wavelength undergoes TIR at this interface."""
    sin2_i = 1.0 - cos_theta_i ** 2
    sin2_t = (n_i / n_t) ** 2 * sin2_i
    return sin2_t >= 1.0
