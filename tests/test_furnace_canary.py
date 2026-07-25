"""T-tier point-probes for module 9 (furnace_canary.py).

Covers the Neumann-series analytic reference (scalar isotropic-furnace
reduction and the general matrix-operator case, cross-checked against the
exact Fredholm solve per Inverse Paper 1's validation-fixture convention),
plus the z-score and ESS helpers T3/T14/T22 lean on (session_log_restir_7's
"first pass z=-16.96 looked biased, ESS~1.02/4 revealed why" pattern).
"""

import sys
import os
from dataclasses import dataclass

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from furnace_canary import (
    neumann_series_reference,
    fredholm_solve_exact,
    z_score,
    effective_sample_size,
)

torch.set_default_dtype(torch.float64)


@dataclass
class FurnaceScene:
    T: object
    L_e: object


def test_scalar_furnace_matches_closed_form_geometric_series():
    # Isotropic furnace: every point sees the same environment, so the
    # Neumann series collapses to a scalar geometric sum with closed form
    # L_e/(1-rho).
    rho, L_e = 0.6, 2.0
    scene = FurnaceScene(T=rho, L_e=L_e)
    truncated = neumann_series_reference(scene, order=200, tol=1e-15)
    closed_form = L_e / (1.0 - rho)
    assert abs(truncated.item() - closed_form) < 1e-12


def test_scalar_furnace_matches_fredholm_solve_exact():
    scene = FurnaceScene(T=0.6, L_e=2.0)
    ref = neumann_series_reference(scene, order=200, tol=1e-15)
    exact = fredholm_solve_exact(scene)
    assert abs(ref.item() - exact.item()) < 1e-12


def test_matrix_operator_matches_fredholm_solve_exact():
    # General (non-isotropic) transport operator, spectral radius < 1 --
    # same convergence cross-check Inverse Paper 1 runs at startup.
    T = torch.tensor([[0.1, 0.2, 0.0], [0.05, 0.15, 0.1], [0.0, 0.1, 0.2]])
    L_e = torch.tensor([1.0, 0.5, 2.0])
    scene = FurnaceScene(T=T, L_e=L_e)

    assert torch.linalg.eigvals(T).abs().max().item() < 1.0  # convergent

    truncated = neumann_series_reference(scene, order=200, tol=1e-15)
    exact = fredholm_solve_exact(scene)
    assert torch.allclose(truncated, exact, atol=1e-10)


def test_tol_stops_the_series_early_without_changing_the_result():
    scene = FurnaceScene(T=0.5, L_e=1.0)
    loose = neumann_series_reference(scene, order=1000, tol=1e-6)
    tight = neumann_series_reference(scene, order=1000, tol=1e-15)
    # Both converged past the tolerance band; a higher order cap on the
    # loose run wouldn't move its answer any further.
    assert abs(loose.item() - tight.item()) < 1e-5


def test_z_score_matches_definition_exactly():
    reference = 10.0
    stderr = 0.5
    assert z_score(reference + 2 * stderr, stderr, reference) == 2.0
    assert z_score(reference - 3 * stderr, stderr, reference) == -3.0
    assert z_score(reference, stderr, reference) == 0.0


def test_ess_is_full_for_uniform_weights_and_one_for_a_single_spike():
    uniform = torch.ones(8)
    assert abs(effective_sample_size(uniform) - 8.0) < 1e-12

    spike = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert abs(effective_sample_size(spike) - 1.0) < 1e-12


def test_ess_flags_near_degenerate_weight_dynamic_range():
    # Session 7 Test 1b's diagnostic signature: a badly-mismatched proposal
    # leaves one dominant draw and near-zero mass elsewhere -- ESS/M
    # collapses toward 1/M even though M draws were nominally taken.
    M = 4
    weights = torch.tensor([1.0, 1e-60, 1e-60, 1e-60])
    ess = effective_sample_size(weights)
    assert ess < 1.1
    assert ess / M < 0.3


if __name__ == "__main__":
    test_scalar_furnace_matches_closed_form_geometric_series()
    test_scalar_furnace_matches_fredholm_solve_exact()
    test_matrix_operator_matches_fredholm_solve_exact()
    test_tol_stops_the_series_early_without_changing_the_result()
    test_z_score_matches_definition_exactly()
    test_ess_is_full_for_uniform_weights_and_one_for_a_single_spike()
    test_ess_flags_near_degenerate_weight_dynamic_range()
    print("all furnace_canary tests passed")
