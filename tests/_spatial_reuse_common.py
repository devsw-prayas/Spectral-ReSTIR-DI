"""Shared scene-building helpers for the T2-T5 spatial-reuse point-probes.

Not a test file itself (no `test_` prefix, not meant to run standalone) --
the 5-pixel fluorescent-surface neighborhood shared across T2 (well-matched),
T3 (support-mismatch stress test), and T4 (elastic baseline), following
`session_log_restir_7_tier4_spatial_reuse_probes.md`. T5 (dispersive hard
cutoff) builds its own thresholds on top of this same grid/light but keeps
its own module.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from spectral_grid import make_grid

GRID = make_grid(lam_min=400.0, lam_max=700.0, oversampling=1.0)
G = torch.tensor([1.0, 1.2, 0.8, 1.5, 0.9])  # per-pixel geometric factor, 5-pixel neighborhood
N_PIXELS = 5
DEST = 2  # destination pixel index, arbitrary interior pixel of the 5


def gaussian(mu, sigma):
    return torch.exp(-0.5 * ((GRID.lam - mu) / sigma) ** 2)


def light_spectrum():
    return gaussian(550.0, 60.0)  # area light L_e(lambda)


def candidate_mass(L_e):
    """Light-importance-only NEE candidate-generation PROBABILITY MASS over grid indices."""
    mass = L_e * GRID.weights
    return mass / mass.sum()


def gen_density(L_e):
    """p_gen(idx) as a density w.r.t. dlambda -- mass(idx)/weights(idx) = L_e(idx)/Z."""
    mass = L_e * GRID.weights
    Z = mass.sum()
    return L_e / Z


def identity_shift(y):
    return y, 1.0


def identity_shift_rows(n=N_PIXELS):
    row = [identity_shift] * n
    return [row for _ in range(n)]


def quadrature_truth(p_hat_dest_values):
    """∫ p_hat_dest(lambda) dlambda via the grid's own trapezoid weights."""
    return (p_hat_dest_values * GRID.weights).sum().item()
