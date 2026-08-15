"""Graphable sweep for G1: "Heterogeneous coupling vs. edge sharpness w,
bounded-as-w->0" -- does the value stay bounded as the edge gets
arbitrarily sharp, or does it blow up?

**Question this answers:** does heterogeneous-medium
species coupling have a volumetric analog of the SURFACE Woodbury/TIR
"lever-arm" effect -- a boundary-crossing SINGULARITY that blows up as some
sharpness parameter `w` shrinks to zero? The surface effect was powered by a
`dc/dA ~ -4.4e4` derivative diverging exactly at a TIR crossing.

**Answer (confirmed here numerically, matching the historical session's own
finding): no.** The heterogeneous coupling channel runs entirely through
TRANSMITTANCE, which depends on a *cumulative* (integrated) concentration --
`d(integral of c1)/dtheta` stays bounded by `Delta_c` regardless of how sharp
the transition is, because integrating a step (however sharp) against a
bounded function never diverges. There is no support boundary in wavelength
space here (unlike the surface TIR case), so there is no `1/sqrt`-type
singular derivative to act as a lever arm.

**Toy model** (self-contained, no `src/` dependency -- this is
differentiable-rendering-flavored sensitivity analysis, not RIS/ReSTIR reuse,
so none of the 9 checklist modules apply; same standalone-toy treatment as
T1/T26): two species along a 1D volumetric path `z in [0,1]`. Species 1's
concentration `c1(z)` is a sigmoid transition (`c1_hi -> c1_lo`) at position
`z0`, width `w`; species 2's concentration `c2` is uniform. Species 2's
"channel" `I2(theta)` is its own emission integrated against the TOTAL
transmittance (both species' absorption), so `theta`-dependence of species
1's absorption band (`mu1a`) or edge position (`z0`) enters `I2` *only*
through shared transmittance -- exactly the coupling mechanism V-D probes.
`dI2/dtheta` is estimated via central finite differences against a dense
quadrature grid (matching the historical session's own FD-vs-quadrature
methodology, `drift <= 3.5e-6` there).

**Two probes:**
1. **Bounded-as-w->0**: `dI2/dmu1a` and `dI2/dz0` at `w in {0.2, 0.02,
   0.005}` stay within a narrow band (no blow-up), and the smallest-`w`
   value closely matches a directly-computed HARD-STEP (`w=0` exactly,
   `torch.where` cutoff, no sigmoid) reference -- confirms smooth convergence
   to the correct limit, not divergence.
2. **Spectral-separation control** (same gating knob as the surface case,
   G1's own historical "150x/800x" finding): moving species 1's absorption
   band far from the sensor wavelength collapses the coupling derivative by
   several orders of magnitude. Fresh parameters here give a ~300x collapse
   (`mu1a` from 20nm to 60nm off the sensor wavelength) -- same mechanism,
   not a literal replay of the historical 150x/800x numbers (those aren't
   recoverable from the session log's own scratch code, per this repo's
   established reconstruction-not-replay convention).
"""

import torch

torch.set_default_dtype(torch.float64)

Z_GRID = torch.linspace(0.0, 1.0, 60_001)

C1_HI, C1_LO = 2.0, 0.1
C2 = 1.0
E2 = 1.0
SIGMA1 = 15.0
MU2A = 620.0
SIGMA2 = 40.0
LAMBDA_S = 620.0

Z0_BASE = 0.4
MU1A_BASE = 600.0  # 20nm off the sensor wavelength -- near/aligned
MU1A_FAR = 560.0  # 60nm off -- spectrally separated control


def _absorption(mu, sigma):
    return torch.exp(torch.tensor(-0.5 * ((LAMBDA_S - mu) / sigma) ** 2))


def compute_I2(mu1a, z0, w):
    """Species 2's channel: own emission integrated against total (species-1
    + species-2) transmittance. theta = (mu1a, z0) enters only through
    species 1's contribution to that shared transmittance."""
    c1_vals = C1_LO + (C1_HI - C1_LO) * torch.sigmoid(-(Z_GRID - z0) / w)
    a1 = _absorption(mu1a, SIGMA1)
    a2 = _absorption(MU2A, SIGMA2)
    cum_c1 = torch.cat([torch.zeros(1), torch.cumulative_trapezoid(c1_vals, Z_GRID)])
    transmittance = torch.exp(-a1 * cum_c1 - C2 * a2 * Z_GRID)
    return torch.trapz(C2 * E2 * transmittance, Z_GRID).item()


def compute_I2_hard_step(mu1a, z0):
    """w=0 limit: literal step function, no sigmoid -- the reference the
    smooth-sigmoid family should converge to as w shrinks."""
    c1_vals = torch.where(Z_GRID < z0, torch.tensor(C1_HI), torch.tensor(C1_LO))
    a1 = _absorption(mu1a, SIGMA1)
    a2 = _absorption(MU2A, SIGMA2)
    cum_c1 = torch.cat([torch.zeros(1), torch.cumulative_trapezoid(c1_vals, Z_GRID)])
    transmittance = torch.exp(-a1 * cum_c1 - C2 * a2 * Z_GRID)
    return torch.trapz(C2 * E2 * transmittance, Z_GRID).item()


def _central_fd(f, x0, h):
    return (f(x0 + h) - f(x0 - h)) / (2 * h)


def _d_dmu1a(w, mu1a=MU1A_BASE, z0=Z0_BASE, h=0.5):
    return _central_fd(lambda m: compute_I2(m, z0, w), mu1a, h)


def _d_dz0(w, mu1a=MU1A_BASE, z0=Z0_BASE, h=1e-3):
    return _central_fd(lambda z: compute_I2(mu1a, z, w), z0, h)


W_SWEEP = (0.2, 0.02, 0.005)


def test_coupling_derivative_bounded_as_w_shrinks():
    dmu_vals = [_d_dmu1a(w) for w in W_SWEEP]
    dz0_vals = [_d_dz0(w) for w in W_SWEEP]

    for vals in (dmu_vals, dz0_vals):
        assert all(abs(v) < 1.0 for v in vals)  # nowhere near blowing up
        # bounded-as-w->0: successive values stay within a narrow band,
        # not diverging as w shrinks by 40x total across the sweep
        assert max(abs(v) for v in vals) / min(abs(v) for v in vals) < 1.5


def test_smallest_w_matches_hard_step_limit():
    dmu_smallest_w = _d_dmu1a(W_SWEEP[-1])
    dmu_hard_step = _central_fd(lambda m: compute_I2_hard_step(m, Z0_BASE), MU1A_BASE, 0.5)
    assert abs(dmu_smallest_w - dmu_hard_step) / abs(dmu_hard_step) < 0.01


def test_spectral_separation_control_collapses_coupling():
    # same gating knob as the surface case (T11's own spectral-separation
    # control): moving species 1's band away from the sensor wavelength
    # collapses the coupling derivative by orders of magnitude.
    d_near = _d_dmu1a(0.02, mu1a=MU1A_BASE)
    d_far = _d_dmu1a(0.02, mu1a=MU1A_FAR)
    collapse_factor = abs(d_near) / abs(d_far)
    assert collapse_factor > 50.0


if __name__ == "__main__":
    test_coupling_derivative_bounded_as_w_shrinks()
    test_smallest_w_matches_hard_step_limit()
    test_spectral_separation_control_collapses_coupling()
    print("all G1 tests passed")
