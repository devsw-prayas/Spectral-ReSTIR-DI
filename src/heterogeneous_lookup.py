"""Heterogeneous local-lookup — checklist module 6.

C(z) / concentration-field lookup for heterogeneous fluorescent media.
Covers A6 (heterogeneous local-lookup consistency trichotomy), feeds
T7-T14.
Depends on: freepath_sampler (module 5).
"""

import torch


def concentration_at(z, field):
    """Local concentration C(z) lookup into a heterogeneous field."""
    raise NotImplementedError


def lookup_trichotomy_case(z, field, thresholds):
    """Classify which of the three A6 consistency cases applies at z."""
    raise NotImplementedError
