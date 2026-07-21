"""Volumetric free-path sampler (delta-tracking stand-in) — checklist module 5.

Covers A7 (free-path firewall corollary, restir_running_notes.md §8): if
z_A, z_B are themselves random (real lambda-coupled free-path process), the
fix-local reuse result from A6 part 3 holds unconditionally regardless of
z's distribution (tower property) — no property of THIS module's sampler
matters for that proof, only that it produces an unbiased free path at all.
`sample_free_path`'s correctness is exactly the thing V-E-1-style harness
tests (T15) check before trusting cross-pixel reuse tests (T16) built on it.

**Position-sampling firewall (A2 scope lemma):** sigma_t(lambda)-driven
position sampling must stay confined to this module and run *before* any
reservoir is touched — never leaked into `ris_reservoir.py` or
`shift_maps.py`.

1D domain convention (matches the toy-probe family in session_log_restir_2
through _4): z is a scalar position; `sigma_t_fn(z)` bakes in whatever
wavelength context is relevant to the caller (this module is
wavelength-agnostic by design — the lambda-coupling lives entirely in the
`sigma_t_fn` closure the caller provides).

Depends on: ris_reservoir (module 1), shift_maps (module 3).
"""

import torch


def sample_free_path(sigma_t_fn, sigma_t_majorant: float, z_start: float, z_end: float, rng: torch.Generator):
    """Delta-tracking free-path sample from `z_start` toward `z_end`.

    `sigma_t_majorant` must upper-bound `sigma_t_fn(z)` for every z between
    `z_start` and `z_end`, or the result is silently biased (caller's
    responsibility — this module does not verify the bound).

    Returns (z, is_real_collision). `is_real_collision=False` means the ray
    reached `z_end` with no genuine collision ("miss").
    """
    direction = 1.0 if z_end >= z_start else -1.0
    span = abs(z_end - z_start)
    traveled = 0.0

    while True:
        u = torch.rand((), generator=rng).item()
        step = -torch.log(torch.tensor(1.0 - u)).item() / sigma_t_majorant
        traveled += step
        if traveled >= span:
            return z_end, False

        z = z_start + direction * traveled
        accept_u = torch.rand((), generator=rng).item()
        if accept_u < sigma_t_fn(z) / sigma_t_majorant:
            return z, True
        # else: null collision, keep tracking


def free_path_pdf(z: float, sigma_t_fn, z_start: float, n_quad: int = 2000) -> float:
    """Analytic-quadrature free-path density p(z) = sigma_t(z) * exp(-optical_depth(z_start, z)).

    For firewall-harness cross-checks (T15/T16) of `sample_free_path`'s
    empirical distribution against ground truth. `n_quad` sets the trapezoid
    resolution of the optical-depth integral.
    """
    zs = torch.linspace(z_start, z, n_quad)
    sigma_vals = torch.tensor([sigma_t_fn(zz.item()) for zz in zs])
    optical_depth = abs(torch.trapz(sigma_vals, zs).item())
    return sigma_t_fn(z) * torch.exp(torch.tensor(-optical_depth)).item()


def miss_probability(sigma_t_fn, z_start: float, z_end: float, n_quad: int = 2000) -> float:
    """Analytic-quadrature P(no collision in [z_start, z_end]) = exp(-optical_depth)."""
    zs = torch.linspace(z_start, z_end, n_quad)
    sigma_vals = torch.tensor([sigma_t_fn(zz.item()) for zz in zs])
    optical_depth = abs(torch.trapz(sigma_vals, zs).item())
    return torch.exp(torch.tensor(-optical_depth)).item()
