"""Shared scene-building helpers for the T17-T20 temporal-reuse point-probes.

Not a test file itself (no `test_` prefix) -- continuous rank-1 fluorophore
family (`p_hat(lambda') = a(lambda')*L_e(lambda')*G`). Kept deliberately
separate from `test_temporal_history.py`'s discrete 3-item toy: that file
exercises module 7's own API surface directly (module-level unit tests),
these T-item files instead probe specific numeric scenarios with a
continuous spectral target, same split as `_spatial_reuse_common.py` vs.
T2/T6/T12-T14.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from spectral_grid import make_grid

GRID = make_grid(lam_min=400.0, lam_max=700.0, oversampling=1.0)


def gaussian(mu, sigma):
    return torch.exp(-0.5 * ((GRID.lam - mu) / sigma) ** 2)


def light_spectrum():
    return gaussian(550.0, 60.0)  # area light L_e(lambda), shared proposal shape


def candidate_mass(L_e):
    mass = L_e * GRID.weights
    return mass / mass.sum()


def gen_density(L_e):
    mass = L_e * GRID.weights
    Z = mass.sum()
    return L_e / Z


def quadrature_truth(p_hat_values):
    return (p_hat_values * GRID.weights).sum().item()
