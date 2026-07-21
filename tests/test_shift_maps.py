"""T-tier point-probes for module 3 (shift_maps.py).

Checks the A1 theorem's per-vertex classification: identity shift at
diffuse/thin_film/fluorescent vertices, genuine Snell-law reshift at
dispersive vertices (cross-checked against Tier-0 snell_jacobian.py /
cauchy_ior.py directly), and non-existence (None, 0.0) at TIR.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cauchy_ior import cos_theta_t
from snell_jacobian import refracted_direction, solid_angle_ratio
from shift_maps import (
    is_reconnection_valid,
    shift_surface,
    shift_volumetric,
)

torch.set_default_dtype(torch.float64)


def test_trivial_vertex_types_are_identity_shift():
    for vertex_type in ("diffuse", "thin_film", "fluorescent"):
        assert is_reconnection_valid(vertex_type)
        y = torch.tensor([0.1, 0.2, 0.3])
        y_out, J = shift_surface(y, vertex_type)
        assert torch.equal(y_out, y)
        assert J == 1.0


def test_volumetric_shift_always_identity():
    y = torch.tensor([1.0, 0.0, 0.0])
    y_out, J = shift_volumetric(y, lam_A=450.0, lam_B=650.0)
    assert torch.equal(y_out, y)
    assert J == 1.0


def test_dispersive_shift_matches_direct_snell_computation():
    assert not is_reconnection_valid("dispersive")

    omega_i = torch.tensor([0.3, 0.0, -0.9539])
    omega_i = omega_i / omega_i.norm()
    n_hat = torch.tensor([0.0, 0.0, 1.0])
    cos_i = -torch.dot(omega_i, n_hat)
    n_i = torch.tensor(1.0)
    n_t_B = torch.tensor(1.52)

    y_out, J = shift_surface(
        omega_i, "dispersive",
        omega_i=omega_i, cos_i=cos_i, n_hat=n_hat, n_i=n_i, n_t_B=n_t_B,
    )

    cos_t_B_expected = cos_theta_t(cos_i, n_i, n_t_B)
    omega_t_expected = refracted_direction(omega_i, cos_i, cos_t_B_expected, n_i, n_t_B, n_hat)
    J_expected = solid_angle_ratio(n_i, n_t_B, cos_i, cos_t_B_expected)

    assert torch.allclose(y_out, omega_t_expected)
    assert torch.isclose(J, J_expected)


def test_dispersive_shift_returns_none_at_tir():
    # steep grazing incidence into a denser->rarer transition triggers TIR
    omega_i = torch.tensor([0.9999, 0.0, -0.0141])
    omega_i = omega_i / omega_i.norm()
    n_hat = torch.tensor([0.0, 0.0, 1.0])
    cos_i = -torch.dot(omega_i, n_hat)
    n_i = torch.tensor(1.52)
    n_t_B = torch.tensor(1.0)

    y_out, J = shift_surface(
        omega_i, "dispersive",
        omega_i=omega_i, cos_i=cos_i, n_hat=n_hat, n_i=n_i, n_t_B=n_t_B,
    )
    assert y_out is None
    assert J == 0.0


if __name__ == "__main__":
    test_trivial_vertex_types_are_identity_shift()
    test_volumetric_shift_always_identity()
    test_dispersive_shift_matches_direct_snell_computation()
    test_dispersive_shift_returns_none_at_tir()
    print("all shift_maps tests passed")
