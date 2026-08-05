"""Point-probe for T21 (addendum_volumetric_temporal_mclamping.md's "VTest 1":
volumetric temporal reuse, clean baseline).

First probe to combine module 7 (`temporal_history.temporal_combine`) with a
genuinely volumetric target -- T17-T20 only ever used the flat rank-1
surface-style target `a(lambda')*L_e(lambda')*G`; T12-T14 combined a
volumetric target with `mis_combine.combine_reservoirs` directly (spatial,
not temporal). This is the first union of the two: single FIXED vertex (no
`C(z)` field -- matches the addendum's own single-pixel scope, static
geometry, `temporal_history.py`'s locked v1 precondition), scalar optical
depth `C` shared by current and history this frame:

    p_hat(lambda') = a(lambda') * L_e(lambda') * exp(-a(lambda')*C) * G

Clean baseline: `C_hist == C_cur` (no gen/eval mismatch at all, so the
Coverage Lemma's `iff` condition holds trivially, same base case as T17)
-- the harder `C_gen != C_eval` sweep is T22's territory (Rao-Blackwell
re-check at worst points), not this probe.

Uses `heterogeneous_lookup.local_target`'s `transmittance_fn` hook in its
rank-1 degenerate form (single species j=0, constant concentration=1),
same reuse-not-reinvent pattern as T12-T14 -- this is purely borrowing the
transmittance term, not exercising A6's rank-k trichotomy.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from ris_reservoir import Reservoir
from temporal_history import temporal_combine
from heterogeneous_lookup import local_target

from _temporal_reuse_common import (
    GRID,
    light_spectrum,
    gaussian,
    candidate_mass,
    gen_density,
    quadrature_truth,
)

torch.set_default_dtype(torch.float64)

L_E = light_spectrum()
A = gaussian(560.0, 50.0)  # fluorophore absorption band
G = 1.3
C = 0.8  # scalar optical depth, shared by current and history (clean baseline)

MASS = candidate_mass(L_E)
P_GEN = gen_density(L_E)
M = 8


def _transmittance(j, lam_prime, c):
    return torch.exp(-A[lam_prime] * c)


def target_pdf_fn(idx):
    value = local_target(
        species_concentration_fn=lambda j, z: 1.0,
        excitation_fn=lambda j, lambda_s: 1.0,
        absorption_fn=lambda j, lam_prime: A[lam_prime],
        emission_fn=lambda lam_prime: L_E[lam_prime],
        j=0,
        lam_prime=idx,
        z=C,
        lambda_s=None,
        transmittance_fn=lambda j, lam_prime, z: _transmittance(j, lam_prime, z),
    )
    return (value * G).item()


_TRANS = torch.exp(-A * C)
TRUTH = quadrature_truth(A * L_E * _TRANS * G)


def _stream_reservoir(rng):
    r = Reservoir()
    for _ in range(M):
        idx = torch.multinomial(MASS, 1, generator=rng).item()
        p_hat = target_pdf_fn(idx)
        p_gen = P_GEN[idx].item()
        accepted = r.update(idx, p_hat / p_gen, rng)
        if accepted:
            r.set_p_hat_gen(p_hat)
    return r


def test_truth_matches_quadrature_of_the_analytic_target():
    expected = (A * L_E * _TRANS * G * GRID.weights).sum().item()
    assert abs(TRUTH - expected) < 1e-9


def test_clean_volumetric_temporal_baseline_is_unbiased():
    N = 40_000
    rng = torch.Generator().manual_seed(2100)

    combined_samples = torch.empty(N)
    for t in range(N):
        r_cur = _stream_reservoir(rng)
        r_hist = _stream_reservoir(rng)
        combined = temporal_combine(r_cur, r_hist, target_pdf_fn, rng)
        combined_samples[t] = combined.wsum

    mean = combined_samples.mean().item()
    stderr = combined_samples.std().item() / (N ** 0.5)
    z = (mean - TRUTH) / stderr
    assert abs(z) < 3.5


if __name__ == "__main__":
    test_truth_matches_quadrature_of_the_analytic_target()
    test_clean_volumetric_temporal_baseline_is_unbiased()
    print("all T21 tests passed")
