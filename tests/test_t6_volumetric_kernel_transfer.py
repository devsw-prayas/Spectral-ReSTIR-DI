"""Point-probe for T6 (rank-1 volumetric harness-validation "kernel-transfer
sanity check").

Historical V-A: a reservoir (winner lambda', RIS weight, M=8 uniform-
proposal candidates, target a(lambda')*L_e(lambda')) generated at pixel A
(sensor 545nm) was reused VERBATIM at pixel B (sensor 575nm) via the
volumetric shift map. Both native-A and reused-at-B estimates matched
quadrature to <0.03% (|z|<0.6) -- confirming A2's claim that rank-1
recorrelation transfers to a volumetric scattering event by inspection: the
target's spectral SHAPE has no sensor-side/context dependence at all (unlike
the dispersive surface case), so verbatim cross-vertex reuse needs no
reweighting beyond a per-vertex scalar. This probe reconstructs that
qualitative result (unbiased at both ends) with fresh concrete parameters,
not the exact historical numbers/seed -- the original band parameters were
already flagged as reconstructed, not recovered, so there is no bit-exact
target to replay.

Unlike T2-T5 (surface, hand-rolled `identity_shift`), this is the first
point-probe to explicitly wire together module 2
(`recorrelation_sampler.joint_target`, the rank-1 RIS target) and module 3
(`shift_maps.shift_volumetric`, the A2 identity shift) through module 1/4's
reservoir+combine machinery, per the "modules 1-4 unlock T1-T11" checkpoint.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from ris_reservoir import Reservoir
from mis_combine import combine_reservoirs
from recorrelation_sampler import joint_target
from shift_maps import shift_volumetric

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
A = gaussian(560.0, 50.0)  # fluorophore absorption band, SAME at every vertex (homogeneous medium)

MASS = candidate_mass(L_E)
P_GEN = gen_density(L_E)


def _target_pdf(i):
    return lambda idx, i=i: joint_target(A[idx], L_E[idx], G[i]).item()


TARGET_FNS = [_target_pdf(i) for i in range(N_PIXELS)]
SHIFT_FNS = [[shift_volumetric] * N_PIXELS for _ in range(N_PIXELS)]

TRUTH = quadrature_truth(A * L_E * G[DEST])


def test_shift_volumetric_is_unconditional_identity():
    y, J = shift_volumetric(torch.tensor(3), lam_A=500.0, lam_B=650.0)
    assert y.item() == 3
    assert J == 1.0


def test_truth_matches_quadrature_of_the_analytic_target():
    expected = (A * L_E * G[DEST] * GRID.weights).sum().item()
    assert abs(TRUTH - expected) < 1e-9


def test_verbatim_reuse_across_vertices_matches_native_unbiased():
    """Direct echo of V-A's literal table: draw a candidate under vertex A's
    own generation distribution, evaluate it natively at A (single-sample IS
    estimator of the A-integral, `wsum`) and -- via `shift_volumetric`'s
    identity map -- reweighted at vertex B against B's target while still
    drawn from A's generation density (single-sample IS estimator of the
    B-integral). Both must be unbiased for the same y-draws.
    """
    N = 80_000
    rng = torch.Generator().manual_seed(606)
    A_IDX, B_IDX = 0, 4  # well-separated vertices in the neighborhood, distinct G

    truth_A = quadrature_truth(A * L_E * G[A_IDX])
    truth_B = quadrature_truth(A * L_E * G[B_IDX])

    native_A = torch.empty(N)
    reused_B = torch.empty(N)
    for t in range(N):
        idx = torch.multinomial(MASS, 1, generator=rng).item()
        p_gen = P_GEN[idx].item()
        y_B, J = shift_volumetric(idx)
        native_A[t] = TARGET_FNS[A_IDX](idx) / p_gen
        reused_B[t] = TARGET_FNS[B_IDX](y_B) * abs(J) / p_gen

    mean_A = native_A.mean().item()
    stderr_A = native_A.std().item() / (N ** 0.5)
    z_A = (mean_A - truth_A) / stderr_A
    assert abs(z_A) < 3.5

    mean_B = reused_B.mean().item()
    stderr_B = reused_B.std().item() / (N ** 0.5)
    z_B = (mean_B - truth_B) / stderr_B
    assert abs(z_B) < 3.5


def _draw_reservoir(i, rng):
    idx = torch.multinomial(MASS, 1, generator=rng).item()
    w = TARGET_FNS[i](idx) / P_GEN[idx].item()
    r = Reservoir()
    r.update(idx, w, rng)
    r.set_p_hat_gen(TARGET_FNS[i](idx))
    return r


def test_combine_is_unbiased_with_meaningful_variance_reduction():
    N = 40_000
    rng = torch.Generator().manual_seed(707)

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
    assert variance_reduction > 2.0  # matched-shape neighborhood -- meaningful reuse gain


if __name__ == "__main__":
    test_shift_volumetric_is_unconditional_identity()
    test_truth_matches_quadrature_of_the_analytic_target()
    test_verbatim_reuse_across_vertices_matches_native_unbiased()
    test_combine_is_unbiased_with_meaningful_variance_reduction()
    print("all T6 tests passed")
