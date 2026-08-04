"""Point-probe for T14 (session_log_restir_8 "Test 3": volumetric spatial
reuse, extreme optical-depth contrast + ESS diagnostic).

Same target family and logistic-`C(z)` construction as T13, tuned to a wider
contrast: vertex cluster `z in {0.10, 0.40, 0.55, 0.70, 0.95}` gives `C(z)`
ranging from ~0.52 (near-transparent, shallow pixel 4) to ~6.91
(near-opaque, deep pixel 0) -- the most adversarial spread in this family,
qualitatively matching the session log's own 0.65-7.24 range (exact numbers
not reproducible, different closed-form `C(z)`, same "reconstruction, not
replay" discipline as T12/T13). Destination = pixel 0 (deep, near-opaque),
the most adversarial choice per the session log.

Each pixel streams M=8 candidates into its own reservoir before spatial
reuse (session log's own convention for all three volumetric spatial tests,
unlike T12/T13's single-candidate simplification) so an ESS diagnostic on
the shallow pixel's reservoir -- the pixel with the largest local-context
(attenuation) mismatch from the deep destination -- can be checked directly,
mirroring T3's Rao-Blackwell-diagnostic discipline
(`furnace_canary.effective_sample_size`, module 9) without needing a
Rao-Blackwellized re-check: the session log reports a HEALTHY ESS here
(~0.575, unlike T3's genuinely degenerate ~0.255), i.e. this probe's job is
to confirm the clean pass isn't hiding a weight-degeneracy artifact, not to
fix one.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from ris_reservoir import Reservoir
from mis_combine import combine_reservoirs
from shift_maps import shift_volumetric
from heterogeneous_lookup import local_target
from furnace_canary import effective_sample_size

from _spatial_reuse_common import (
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

Z = torch.tensor([0.10, 0.40, 0.55, 0.70, 0.95])  # extreme contrast, Test 3's regime
DEST = 0  # deep, near-opaque pixel -- most adversarial destination per the session log
SHALLOW = 4  # near-transparent pixel -- largest local-context mismatch from DEST

C0, C1, K, Z0 = 0.5, 10.0, 12.0, 0.35  # logistic concentration, wider contrast than T13

M = 8  # candidates streamed per pixel, matching session_log_restir_8's own convention


def _softplus(x: float) -> float:
    return torch.nn.functional.softplus(torch.tensor(x)).item()


def optical_depth(z: float) -> float:
    return C0 * (1.0 - z) + (C1 / K) * (_softplus(K * (1.0 - Z0)) - _softplus(K * (z - Z0)))


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


def _stream_pixel(i, rng, record=False):
    r = Reservoir()
    draws = []
    for _ in range(M):
        idx = torch.multinomial(MASS, 1, generator=rng).item()
        p_hat = TARGET_FNS[i](idx)
        p_gen = P_GEN[idx].item()
        w = p_hat / p_gen
        accepted = r.update(idx, w, rng)
        if accepted:
            r.set_p_hat_gen(p_hat)
        if record:
            draws.append((idx, w))
    return r, draws


def test_extreme_contrast_spans_near_transparent_to_near_opaque():
    c_shallow = optical_depth(Z[SHALLOW].item())
    c_deep = optical_depth(Z[DEST].item())
    assert c_shallow < 1.0
    assert c_deep > 5.0


def test_shallow_pixel_reservoir_ess_is_healthy_not_degenerate():
    rng = torch.Generator().manual_seed(1400)
    _, draws = _stream_pixel(SHALLOW, rng, record=True)
    weights = torch.tensor([w for _, w in draws])
    ess = effective_sample_size(weights)
    assert ess / M > 0.3  # healthy, unlike T3's genuinely degenerate ~0.255


def test_combine_is_unbiased_at_extreme_contrast():
    N = 20_000
    rng = torch.Generator().manual_seed(1401)

    combined_samples = torch.empty(N)
    baseline_samples = torch.empty(N)
    for t in range(N):
        reservoirs = [_stream_pixel(i, rng)[0] for i in range(N_PIXELS)]
        combined = combine_reservoirs(reservoirs, TARGET_FNS, SHIFT_FNS, dest_index=DEST, rng=rng)
        combined_samples[t] = combined.wsum
        # M candidates were streamed into this reservoir (unlike combine_reservoirs'
        # single-candidate-per-source treatment), so wsum estimates M*TRUTH -- the
        # per-frame unbiased single-sample estimate is wsum/M (= contribution_weight()
        # times p_hat_gen(y), matching the module 1 docstring's W definition).
        baseline_samples[t] = reservoirs[DEST].wsum / M

    combined_mean = combined_samples.mean().item()
    combined_stderr = combined_samples.std().item() / (N ** 0.5)
    z_combined = (combined_mean - TRUTH) / combined_stderr
    assert abs(z_combined) < 3.5

    baseline_mean = baseline_samples.mean().item()
    baseline_stderr = baseline_samples.std().item() / (N ** 0.5)
    z_baseline = (baseline_mean - TRUTH) / baseline_stderr
    assert abs(z_baseline) < 3.5  # sanity: no-reuse baseline is also unbiased on its own


if __name__ == "__main__":
    test_extreme_contrast_spans_near_transparent_to_near_opaque()
    test_shallow_pixel_reservoir_ess_is_healthy_not_degenerate()
    test_combine_is_unbiased_at_extreme_contrast()
    print("all T14 tests passed")
