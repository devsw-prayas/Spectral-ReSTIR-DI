import torch


def refracted_direction(
    omega_i: torch.Tensor,
    cos_i:   torch.Tensor,
    cos_t:   torch.Tensor,
    n_i,
    n_t,
    n_hat:   torch.Tensor,
) -> torch.Tensor:
    """Vector Snell's law: compute transmitted direction ω_t.

    Convention (used throughout this module):
        omega_i points TOWARD the surface (ray travel direction).
        n_hat   points TOWARD the incident medium (against the incoming ray).
        cos_i   = -dot(omega_i, n_hat) > 0.
        omega_t points AWAY from the surface (into transmitted medium).

    omega_i: (..., 3)
    cos_i:   (...) or (N,)
    cos_t:   (...) or (N,)  — from cauchy_ior.cos_theta_t()
    n_i, n_t: scalar or (N,)
    n_hat:   (3,)
    Returns: same leading shape as omega_i, last dim 3.
    """
    eta = n_i / n_t
    return eta * omega_i + (eta * cos_i - cos_t).unsqueeze(-1) * n_hat


def snell_jacobian(
    n_i:   torch.Tensor,
    n_t:   torch.Tensor,
    cos_i: torch.Tensor,
    cos_t: torch.Tensor,
    n_hat: torch.Tensor,
) -> torch.Tensor:
    """∂ω_t/∂ω_i — Lemma 1 Snell Jacobian.

    J = (n_i/n_t) [I₃ − (1 − n_i cosθ_i / (n_t cosθ_t)) n̂n̂ᵀ]

    Diverges as cosθ_t → 0 (TIR onset). Call snell_jacobian_tir_safe()
    for wavelengths near or at the critical angle.

    n_i, n_t: (N,)
    cos_i:    (N,) or broadcastable scalar
    cos_t:    (N,)  — must not be zero (no TIR wavelengths)
    n_hat:    (3,)
    Returns:  (N, 3, 3)
    """
    N = n_i.shape[0]
    ratio = n_i / n_t                                           # (N,)
    c     = 1.0 - ratio * cos_i / cos_t                        # (N,)

    I3  = torch.eye(3, dtype=n_i.dtype, device=n_i.device)
    nnT = torch.outer(n_hat, n_hat)                             # (3, 3)

    J = ratio.view(N,1,1) * (
        I3.unsqueeze(0) - c.view(N,1,1) * nnT.unsqueeze(0)
    )
    return J                                                    # (N, 3, 3)


def solid_angle_ratio(
    n_i:   torch.Tensor,
    n_t:   torch.Tensor,
    cos_i: torch.Tensor,
    cos_t: torch.Tensor,
) -> torch.Tensor:
    """Solid angle ratio |dω_t / dω_i| = (n_i/n_t)² cosθ_i / cosθ_t.

    This is the 2-D Jacobian of the sphere map ω_i → ω_t (eigenvalue of the
    tangential block of the 3×3 Snell Jacobian, squared and divided by the
    normal eigenvalue).

    Diverges at TIR (cosθ_t → 0). Returns (N,).
    """
    return (n_i / n_t) ** 2 * cos_i / cos_t


def tir_jacobian(
    v:     torch.Tensor,
    n_i:   torch.Tensor,
    n_t:   torch.Tensor,
    cos_i: torch.Tensor,
    polarization: str = "unpolarized",
) -> torch.Tensor:
    """TIR-safe scalar J_TIR(v) = f_BTDF(v) · η²c/v  (Theorem 3).

    f_BTDF(v) = T_Fresnel(v) / η² is the physically complete BTDF throughput,
    including the n²-law radiance-compression factor across the refracting
    interface (Veach 1997 §5.2 non-symmetric scattering) — not bare Fresnel
    power transmittance. The η² cancels exactly against the solid-angle det's
    own η² (see solid_angle_ratio), leaving J_TIR = T_Fresnel(v) · c/v.

    The 0×∞ at v = cosθ_t → 0 resolves analytically to a finite limit:
        J_TIR^s(0) = 4/η,   J_TIR^p(0) = 4η

    Closed forms (real-analytic at v = 0):
        J_TIR^s(v) = 4ηc² / (ηc + v)²
        J_TIR^p(v) = 4ηc² / (c + ηv)²

    Derived from T_s = 4ηcv/(ηc+v)² and T_p = 4ηcv/(c+ηv)², divided by η²
    (radiance compression) and multiplied by the solid-angle det η²c/v — the
    v and one power of η² cancel exactly, leaving rational forms.

    v:     (N,)  cosθ_t, must be ≥ 0 (TIR wavelengths should be masked by caller)
    n_i, n_t, cos_i: (N,)
    Returns (N,).
    """
    eta = n_i / n_t
    c   = cos_i
    J_s = 4.0 * eta * c ** 2 / (eta * c + v) ** 2
    J_p = 4.0 * eta * c ** 2 / (c + eta * v) ** 2
    if polarization == "s":
        return J_s
    elif polarization == "p":
        return J_p
    return 0.5 * (J_s + J_p)


def snell_jacobian_tir_safe(
    v:     torch.Tensor,
    n_i:   torch.Tensor,
    n_t:   torch.Tensor,
    n_hat: torch.Tensor,
) -> torch.Tensor:
    """TIR-safe combined factor F(v) = J(v) · |∂cosθ_i/∂v|, substitution v = cosθ_t.

    Theorem 3: substituting v = cosθ_t into the gradient integral removes the
    1/cosθ_t singularity. The combined factor is:

        F(v) = α (I − n̂n̂ᵀ) + β n̂n̂ᵀ
        η     = n_i / n_t
        α     = v / (η · cosθ_i(v))      tangential eigenvalue
        β     = 1                         normal eigenvalue (singularities cancel)
        cosθ_i(v) = sqrt(n_i² − n_t²(1 − v²)) / n_i

    Derivation: J tangential = η, J normal = η² cosθ_i/v,
    |∂cosθ_i/∂v| = v/(η² cosθ_i); multiplying gives α and β above.

    At v = 0 (TIR onset):  F(0) = n̂n̂ᵀ  — finite.

    v:         (N,) cosθ_t values in [0, 1]
    n_i, n_t:  (N,)
    n_hat:     (3,)
    Returns:   (N, 3, 3)
    """
    N   = n_i.shape[0]
    eta = n_i / n_t                                             # (N,)

    # cosθ_i as a function of v = cosθ_t via Snell: n_i sinθ_i = n_t sinθ_t
    cos_i_v = torch.sqrt(
        (n_i**2 - n_t**2 * (1.0 - v**2)).clamp(min=0.0)
    ) / n_i                                                     # (N,)

    # α = v / (η cosθ_i(v)); numerator → 0 at v=0, denominator finite → α(0) = 0
    alpha = v / (eta * cos_i_v).clamp(min=1e-30)               # (N,)
    beta  = torch.ones(N, dtype=n_i.dtype, device=n_i.device)  # (N,)

    I3    = torch.eye(3, dtype=n_i.dtype, device=n_i.device)
    nnT   = torch.outer(n_hat, n_hat)                           # (3, 3)
    ImnnT = I3 - nnT

    F = (alpha.view(N,1,1) * ImnnT.unsqueeze(0)
       + beta.view(N,1,1)  * nnT.unsqueeze(0))
    return F                                                    # (N, 3, 3)


def propagate_velocity(V: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """Apply Snell Jacobian to a path velocity field V at one interface.

    V: (N, 3) or (3,) velocity ∂x/∂θ at each wavelength
    J: (N, 3, 3)
    Returns: (N, 3)  V_out[k] = J[k] @ V[k]
    """
    if V.dim() == 1:
        V = V.unsqueeze(0).expand(J.shape[0], -1)
    return torch.einsum("nij,nj->ni", J, V)


def compose_jacobians(Js: list) -> torch.Tensor:
    """Compose per-vertex Snell Jacobians for a multi-bounce path.

    Js: [J_0, J_1, ..., J_{K-1}], each (N, 3, 3)
        J_0 is the first refractive interface the ray hits.

    Returns J_{K-1} @ ... @ J_1 @ J_0  (N, 3, 3).

    Cross-couples all wavelengths because n(λ) differs at each vertex.
    """
    result = Js[0]
    for J in Js[1:]:
        result = torch.bmm(J, result)
    return result
