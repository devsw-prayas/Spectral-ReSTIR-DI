"""Point-probe for T12 (session_log_restir_8 "Test 1": volumetric spatial
reuse, well-matched vertex neighborhood).

First point-probe to combine THREE things T6 kept separate: a genuinely
volumetric target with a position-dependent transmittance term (via
`heterogeneous_lookup.local_target`'s `transmittance_fn` hook, module 6),
`shift_maps.shift_volumetric`'s identity shift (module 3), and
`mis_combine.combine_reservoirs` across a 5-pixel neighborhood (module 4) --
T6 only ever verified verbatim single-candidate reuse, never a full spatial
combine, and never a transmittance-coupled target. Rank-1 (single
fluorophore species, `heterogeneous_lookup`'s species machinery degenerates
to a constant j=0 with concentration 1 -- this is NOT the rank-k trichotomy
of T7-T11, just reusing the module's `transmittance_fn` hook for its
intended volumetric purpose).

Model (session_log_restir_8): single rank-1 fluorophore, absorption `a(lambda')`
and light `L_e(lambda')` both Gaussian. Each pixel has its own FIXED vertex
position `z_i` (free-path sampling already landed there this frame --
`freepath_sampler.py`/module 5 is NOT invoked here, same as the session log's
own framing: "representing where free-path sampling already landed for that
pixel this frame"). Per-pixel target:

    p_hat(z_i, lambda') = a(lambda') * L_e(lambda') * exp(-a(lambda')*C(z_i)) * G_i

`C(z)` is the optical depth from vertex `z` to the light, `C(z) = integral_z^1
c(z') dz'` for a position-dependent concentration field `c(z')`. Session log
uses a logistic/softplus concentration for a closed-form antiderivative; this
probe uses a LINEAR concentration `c(z') = c0 + c1*z'` instead (closed form
`C(z) = c0*(1-z) + c1*(1-z**2)/2`, verified against quadrature below) -- the
concentration shape isn't the content under test (A2/A6/A8 don't care), only
that `C(z)` is analytic and genuinely position-dependent, so no need to
replicate the session log's exact functional form (same "reconstruction, not
replay" discipline as T1/T6/T7-T11 -- the original scratch numbers are lost).

Test 1's specific regime: tight vertex cluster
`z in {0.42, 0.45, 0.48, 0.50, 0.53}` (mild C(z) spread, no steep transition
-- that stress case is T13). Candidate generation: light-importance-only
`q(lambda') ~ L_e(lambda')`, matching every other spatial-reuse probe in this
family (T2/T6). Destination = pixel 2 (`_spatial_reuse_common.DEST`).
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
    DEST,
    light_spectrum,
    gaussian,
    candidate_mass,
    gen_density,
    quadrature_truth,
)

torch.set_default_dtype(torch.float64)

L_E = light_spectrum()
A = gaussian(560.0, 50.0)  # fluorophore absorption band, same at every vertex (homogeneous medium)

Z = torch.tensor([0.42, 0.45, 0.48, 0.50, 0.53])  # tight vertex cluster, Test 1's regime

C0, C1 = 0.5, 1.0  # linear concentration c(z') = C0 + C1*z'


def optical_depth(z: float) -> float:
    """Analytic C(z) = integral_z^1 (C0 + C1*z') dz' for the linear concentration field."""
    return C0 * (1.0 - z) + C1 * (1.0 - z ** 2) / 2.0


def _quadrature_optical_depth(z: float, n_quad: int = 4000) -> float:
    zs = torch.linspace(z, 1.0, n_quad)
    c_vals = C0 + C1 * zs
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

_TRANS_DEST = torch.exp(-A * optical_depth(Z[DEST].item()))
TRUTH = quadrature_truth(A * L_E * _TRANS_DEST * G[DEST])


def test_optical_depth_matches_quadrature():
    for z in (0.05, 0.42, 0.5, 0.9, 1.0):
        analytic = optical_depth(z)
        numeric = _quadrature_optical_depth(z)
        assert abs(analytic - numeric) < 1e-9


def test_shift_volumetric_is_unconditional_identity():
    y, J = shift_volumetric(torch.tensor(3), lam_A=500.0, lam_B=650.0)
    assert y.item() == 3
    assert J == 1.0


def test_truth_matches_quadrature_of_the_analytic_target():
    expected = (A * L_E * _TRANS_DEST * G[DEST] * GRID.weights).sum().item()
    assert abs(TRUTH - expected) < 1e-9


def _draw_reservoir(i, rng):
    idx = torch.multinomial(MASS, 1, generator=rng).item()
    w = TARGET_FNS[i](idx) / P_GEN[idx].item()
    r = Reservoir()
    accepted = r.update(idx, w, rng)
    if accepted:
        r.set_p_hat_gen(TARGET_FNS[i](idx))
    return r


def test_combine_is_unbiased_with_meaningful_variance_reduction():
    N = 40_000
    rng = torch.Generator().manual_seed(1208)

    combined_samples = torch.empty(N)
    baseline_samples = torch.empty(N)
    for t in range(N):
        reservoirs = [_draw_reservoir(i, rng) for i in range(N_PIXELS)]
        combined = combine_reservoirs(reservoirs, TARGET_FNS, SHIFT_FNS, dest_index=DEST, rng=rng)
        combined_samples[t] = combined.wsum
        baseline_samples[t] = reservoirs[DEST].wsum

    combined_mean = combined_samples.mean().item()
    combined_stderr = combined_samples.std().item() / (N ** 0.5)
    z_combined = (combined_mean - TRUTH) / combined_stderr
    assert abs(z_combined) < 3.5

    baseline_mean = baseline_samples.mean().item()
    baseline_stderr = baseline_samples.std().item() / (N ** 0.5)
    z_baseline = (baseline_mean - TRUTH) / baseline_stderr
    assert abs(z_baseline) < 3.5  # sanity: no-reuse baseline is also unbiased on its own

    variance_reduction = baseline_samples.var().item() / combined_samples.var().item()
    assert variance_reduction > 2.0  # tight vertex cluster, mild C(z) spread -- meaningful reuse gain


if __name__ == "__main__":
    test_optical_depth_matches_quadrature()
    test_shift_volumetric_is_unconditional_identity()
    test_truth_matches_quadrature_of_the_analytic_target()
    test_combine_is_unbiased_with_meaningful_variance_reduction()
    print("all T12 tests passed")
