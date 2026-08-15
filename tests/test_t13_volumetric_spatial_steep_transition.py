"""Point-probe for T13 ("Test 2": volumetric spatial reuse, vertices spread
across a steep C(z) transition, two destinations).

Same target family as T12 (`p_hat(z_i, lambda') = a(lambda')*L_e(lambda')*
exp(-a(lambda')*C(z_i))*G_i`, `heterogeneous_lookup.local_target`'s
`transmittance_fn` hook + `shift_maps.shift_volumetric` identity +
`mis_combine.combine_reservoirs`), but T12's mild linear `C(z)` can't produce
a genuine steep-transition stress case (no curvature to be steep with). This
probe swaps in a logistic concentration field `c(z') = C0 + C1*sigmoid(K*(z'-Z0))`
instead, whose closed-form antiderivative is a softplus term -- still exactly
analytic (no raw grid FD against sharp structure, same discipline as the
session log's own choice, just a different closed form since the original
scratch numbers are lost, per T1/T6/T7-T11's "reconstruction, not replay").

Vertex cluster `z in {0.05, 0.35, 0.50, 0.65, 0.90}` spans the steepest part
of the transition (centered at Z0=0.5) plus one deep/far vertex, so local
optical contexts (attenuation to light) genuinely diverge pixel-to-pixel --
the volumetric analog of a surface support-mismatch stress case, except the
mismatch here comes from position-dependent attenuation, not a spectral tint.
Session log tests two destinations (deep/far pixel and mid-transition pixel);
this probe does the same: pixel 0 (deep, z=0.05) and pixel 2 (mid-transition,
z=0.50).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from ris_reservoir import Reservoir
from mis_combine import combine_reservoirs
from shift_maps import shift_volumetric
from heterogeneous_lookup import local_target

from _spatial_reuse_common import (
    GRID,
    G,
    N_PIXELS,
    light_spectrum,
    gaussian,
    candidate_mass,
    gen_density,
    quadrature_truth,
)

torch.set_default_dtype(torch.float64)

L_E = light_spectrum()
A = gaussian(560.0, 50.0)  # fluorophore absorption band, same at every vertex (homogeneous medium)

Z = torch.tensor([0.05, 0.35, 0.50, 0.65, 0.90])  # spans the steep transition, Test 2's regime

C0, C1, K, Z0 = 0.3, 3.0, 15.0, 0.5  # logistic concentration c(z') = C0 + C1*sigmoid(K*(z'-Z0))


def _softplus(x: float) -> float:
    return torch.nn.functional.softplus(torch.tensor(x)).item()


def optical_depth(z: float) -> float:
    """Analytic C(z) = integral_z^1 (C0 + C1*sigmoid(K*(z'-Z0))) dz'.

    integral sigmoid(K*(z'-Z0)) dz' = softplus(K*(z'-Z0)) / K.
    """
    return C0 * (1.0 - z) + (C1 / K) * (_softplus(K * (1.0 - Z0)) - _softplus(K * (z - Z0)))


def _quadrature_optical_depth(z: float, n_quad: int = 8000) -> float:
    zs = torch.linspace(z, 1.0, n_quad)
    c_vals = C0 + C1 * torch.sigmoid(K * (zs - Z0))
    return torch.trapz(c_vals, zs).item()


MASS = candidate_mass(L_E)
P_GEN = gen_density(L_E)


def _transmittance(j, lam_prime, z):
    return torch.exp(-A[lam_prime] * optical_depth(z))


def _target_pdf(i):
    z_i = Z[i].item()

    def f(idx, z_i=z_i, i=i):
        value = local_target(
            species_concentration_fn=lambda j, z: 1.0,
            excitation_fn=lambda j, lambda_s: 1.0,
            absorption_fn=lambda j, lam_prime: A[lam_prime],
            emission_fn=lambda lam_prime: L_E[lam_prime],
            j=0,
            lam_prime=idx,
            z=z_i,
            lambda_s=None,
            transmittance_fn=lambda j, lam_prime, z: _transmittance(j, lam_prime, z),
        )
        return (value * G[i]).item()

    return f


TARGET_FNS = [_target_pdf(i) for i in range(N_PIXELS)]
SHIFT_FNS = [[shift_volumetric] * N_PIXELS for _ in range(N_PIXELS)]


def _draw_reservoir(i, rng):
    idx = torch.multinomial(MASS, 1, generator=rng).item()
    w = TARGET_FNS[i](idx) / P_GEN[idx].item()
    r = Reservoir()
    accepted = r.update(idx, w, rng)
    if accepted:
        r.set_p_hat_gen(TARGET_FNS[i](idx))
    return r


def test_optical_depth_matches_quadrature():
    for z in (0.05, 0.35, 0.50, 0.65, 0.90, 1.0):
        analytic = optical_depth(z)
        numeric = _quadrature_optical_depth(z)
        assert abs(analytic - numeric) < 1e-6


def _truth_for(dest):
    trans_dest = torch.exp(-A * optical_depth(Z[dest].item()))
    return quadrature_truth(A * L_E * trans_dest * G[dest])


def _run_combine_for_destination(dest, seed):
    truth = _truth_for(dest)
    N = 40_000
    rng = torch.Generator().manual_seed(seed)

    combined_samples = torch.empty(N)
    baseline_samples = torch.empty(N)
    for t in range(N):
        reservoirs = [_draw_reservoir(i, rng) for i in range(N_PIXELS)]
        combined = combine_reservoirs(reservoirs, TARGET_FNS, SHIFT_FNS, dest_index=dest, rng=rng)
        combined_samples[t] = combined.wsum
        baseline_samples[t] = reservoirs[dest].wsum

    combined_mean = combined_samples.mean().item()
    combined_stderr = combined_samples.std().item() / (N ** 0.5)
    z_combined = (combined_mean - truth) / combined_stderr

    baseline_mean = baseline_samples.mean().item()
    baseline_stderr = baseline_samples.std().item() / (N ** 0.5)
    z_baseline = (baseline_mean - truth) / baseline_stderr

    return z_combined, z_baseline


def test_combine_is_unbiased_destination_pixel0_deep_far():
    z_combined, z_baseline = _run_combine_for_destination(dest=0, seed=1300)
    assert abs(z_combined) < 3.5
    assert abs(z_baseline) < 3.5  # sanity: no-reuse baseline is also unbiased on its own


def test_combine_is_unbiased_destination_pixel2_mid_transition():
    z_combined, z_baseline = _run_combine_for_destination(dest=2, seed=1301)
    assert abs(z_combined) < 3.5
    assert abs(z_baseline) < 3.5


if __name__ == "__main__":
    test_optical_depth_matches_quadrature()
    test_combine_is_unbiased_destination_pixel0_deep_far()
    test_combine_is_unbiased_destination_pixel2_mid_transition()
    print("all T13 tests passed")
