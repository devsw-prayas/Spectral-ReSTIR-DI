"""Shift maps T_{lambda_A -> lambda_B} — checklist module 3.

Reconnection-validity machinery for A1 (surface, restir_running_notes.md §2)
and A2 (volumetric, §3). The theorem: T≡id, J≡1 is valid **iff** p(y|lambda)
is literally independent of lambda (support included, not just in
expectation).

Per-vertex classification (locked v1 scope: elastic + rank-1 fluorescent
surface, rank-1 fluorescent volumetric — see restir_running_notes.md §0):
    diffuse      — trivial, Y_lambda == Y for every lambda
    thin_film    — trivial, kernel re-evaluation only, not a measure change
    fluorescent  — trivial AND stronger: zero statistical dependence via the
                   Recorrelation Lemma (recorrelation_sampler.py), not just
                   no support change
    dispersive   — nontrivial: Y_lambda genuinely varies via Snell's
                   transmission cone. Uses Tier-0 snell_jacobian.py /
                   cauchy_ior.py. Near/at TIR the map isn't even onto — an
                   existence failure no Jacobian repairs (forces fallback to
                   collapse+MIS upstream, not handled in this module).

Volumetric (A2) locked scope: isotropic/HG/Rayleigh phase functions (angular
shape lambda-independent) + rank-1 fluorescent scattering satisfy both of
A2's premises unconditionally, so the volumetric shift is always identity.
sigma_t(lambda)'s position-dependence is firewalled to the free-path sampler
(module 5), which runs *before* the reservoir is touched (A2's
position-sampling firewall scope lemma) — never a shift-map concern.

Depends on: ris_reservoir (module 1), snell_jacobian / cauchy_ior (Tier 0).
"""

import torch

from cauchy_ior import cos_theta_t, is_tir
from snell_jacobian import refracted_direction, solid_angle_ratio

RECONNECTION_TRIVIAL_TYPES = ("diffuse", "thin_film", "fluorescent")
RECONNECTION_NONTRIVIAL_TYPES = ("dispersive",)


def is_reconnection_valid(vertex_type: str) -> bool:
    """True iff T≡id, J≡1 at this surface vertex type (A1 theorem)."""
    if vertex_type in RECONNECTION_TRIVIAL_TYPES:
        return True
    if vertex_type in RECONNECTION_NONTRIVIAL_TYPES:
        return False
    raise ValueError(f"unknown vertex_type: {vertex_type!r}")


def shift_surface(
    y,
    vertex_type: str,
    omega_i: torch.Tensor | None = None,
    cos_i: torch.Tensor | None = None,
    n_hat: torch.Tensor | None = None,
    n_i=None,
    n_t_B=None,
):
    """Surface shift map T_{lambda_A -> lambda_B}(y) and its Jacobian |J| (A1).

    diffuse / thin_film / fluorescent: identity shift, J=1 — A1's forward
    direction, p(y|lambda) is literally lambda-independent so no correction
    is possible to omit because none is required.

    dispersive: `y` is a direction omega_i on the incident side; genuinely
    reshifted to the wavelength-B transmission direction via Snell's law.
    Returns (None, 0.0) at TIR — the map doesn't exist there, signaling the
    caller to fall back (collapse+MIS), not to trust an identity shift.
    """
    if vertex_type in RECONNECTION_TRIVIAL_TYPES:
        return y, 1.0
    if vertex_type not in RECONNECTION_NONTRIVIAL_TYPES:
        raise ValueError(f"unknown vertex_type: {vertex_type!r}")

    if is_tir(cos_i, n_i, n_t_B):
        return None, 0.0

    cos_t_B = cos_theta_t(cos_i, n_i, n_t_B)
    omega_t_B = refracted_direction(omega_i, cos_i, cos_t_B, n_i, n_t_B, n_hat)
    J = solid_angle_ratio(n_i, n_t_B, cos_i, cos_t_B)
    return omega_t_B, J


def shift_volumetric(y, lam_A=None, lam_B=None):
    """Volumetric shift map (A2) — always identity in locked v1 scope.

    Both A2 premises (lambda-independent phase-function angular shape,
    rank-1 fluorescent scattering) hold unconditionally for this project's
    scope, so T≡id, J≡1 always — see module docstring.
    """
    return y, 1.0


def shift_jacobian(y, vertex_type: str, **kwargs) -> float:
    """Convenience wrapper returning just |J| from `shift_surface`."""
    _, J = shift_surface(y, vertex_type, **kwargs)
    return J
