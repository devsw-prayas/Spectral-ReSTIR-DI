"""Furnace-canary / Neumann-series analytic reference -- checklist module 9.

Analytic reference for V-tier real-tracer verification, and for T-tier
z-score/ESS/Rao-Blackwell checks (T3, T14, T22).
Depends on: heterogeneous_lookup (module 6), temporal_history (module 7).

The furnace test + closed-form Neumann reference is "shared wholesale" from
Inverse Paper 1 (`session_log_restir_1_planning.md` sec 0) -- this mirrors
that repo's `forward.py` (`neumann_forward`/`fredholm_solve_exact`)
one-for-one: `L = L_e + T@L_e + T^2@L_e + ...` truncated to a bounce depth,
generalized here to accept either a full transport operator (square tensor,
matches the sibling repo's spectral case) or a scalar reflectance `rho`
(the literal furnace-canary reduction, where every point sees the same
isotropic environment and the series collapses to a scalar geometric sum,
closed form `L_e/(1-rho)`).
"""

import torch


def neumann_series_reference(scene, order: int, tol: float):
    """Closed-form Neumann-series radiance reference for a furnace-canary scene.

    `scene` exposes `.T` (transport operator: square `(N, N)` tensor, or a
    scalar/0-d tensor for the isotropic-furnace reduction) and `.L_e`
    (source: `(N,)` tensor or scalar). Sums `L_e + T@L_e + T^2@L_e + ...`
    up to `order` bounces, stopping early once a bounce's contribution's
    norm drops below `tol` (the geometric-series tail is then negligible --
    `order` is a hard cap, `tol` is the adaptive early-exit).
    """
    T = torch.as_tensor(scene.T)
    L_e = torch.as_tensor(scene.L_e)

    L = L_e.clone()
    term = L_e.clone()
    for _ in range(order):
        term = T @ term if T.ndim == 2 else T * term
        L = L + term
        if term.norm() < tol:
            break
    return L


def fredholm_solve_exact(scene):
    """Exact infinite-bounce solve `(I - T) L = L_e` -- validation fixture only.

    Mirrors Inverse Paper 1's `fredholm_solve_exact`: used to confirm a
    scene's spectral radius < 1 (series convergent) and cross-check
    `neumann_series_reference` at high `order`, not as the estimator itself.
    """
    T = torch.as_tensor(scene.T)
    L_e = torch.as_tensor(scene.L_e)
    if T.ndim == 2:
        N = T.shape[0]
        A = torch.eye(N, dtype=T.dtype) - T
        return torch.linalg.solve(A, L_e)
    return L_e / (1.0 - T)


def z_score(estimate, estimate_stderr, reference):
    """Z-score of an MC estimate against the analytic reference."""
    return (estimate - reference) / estimate_stderr


def effective_sample_size(weights):
    """ESS = (sum w)^2 / sum(w^2), for Rao-Blackwell / reuse-quality checks."""
    w = torch.as_tensor(weights, dtype=torch.get_default_dtype())
    return (w.sum() ** 2 / (w ** 2).sum()).item()
