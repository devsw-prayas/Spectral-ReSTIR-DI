"""Furnace-canary / Neumann-series analytic reference — checklist module 9.

Analytic reference for V-tier real-tracer verification, and for T-tier
z-score/ESS/Rao-Blackwell checks (T3, T14, T22).
Depends on: heterogeneous_lookup (module 6), temporal_history (module 7).
"""

import torch


def neumann_series_reference(scene, order, tol):
    """Closed-form Neumann-series radiance reference for a furnace-canary scene."""
    raise NotImplementedError


def z_score(estimate, estimate_stderr, reference):
    """Z-score of an MC estimate against the analytic reference."""
    raise NotImplementedError


def effective_sample_size(weights):
    """ESS = (sum w)^2 / sum(w^2), for Rao-Blackwell / reuse-quality checks."""
    raise NotImplementedError
