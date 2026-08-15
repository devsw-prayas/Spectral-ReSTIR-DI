"""Point-probe for T27 (composition lemma validated against the exact
Bitterli/Veach single-vertex reconnection Jacobian, not an abstract 1D
reprojection map).

T26 validated the composition lemma with an abstract 1D deformation
`phi(x)` and `J_reproj = phi'(x)`. T27 replaces the abstract map with the
actual literature formula for reusing a FIXED world-space point `y` (a
point light) across two DIFFERENT shading points `x1 -> x2`:

    J_reproj = [cos_theta_o(x2) / cos_theta_o(x1)] * [d(x1,y)^2 / d(x2,y)^2]

where `G(x) = cos_theta_o(x) / d(x,y)^2` is the point-light geometric term.
By construction `G(x1) * J_reproj == G(x2)` algebraically -- this is checked
directly as a floating-point implementation-correctness probe (two
independently-computed quantities, `G(x1)*J_reproj` via the ratio formula
and `G(x2)` via its own definition, must agree to machine precision), same
role and same expected ~1e-16 tightness as the historical session's own
3.9e-16 result.

**What plays the role of T26's "candidate generation domain":** unlike T26
(uniform-in-parameter proposal), here the more physically realistic setup
is that a history reservoir's own RIS process already importance-sampled
its candidates proportional to `G(x1)` (the point-light NEE target shape at
the ORIGINAL shading point) -- so candidates `s` are drawn via rejection
sampling proportional to `G1(s) = G(x1(s))`, matching the same "candidates
drawn via rejection sampling proportional to the source Jacobian/target"
technique T28 (section 1.3's 2D-surface probe) and the historical T1.3 both
use. `J_reproj = G2(s)/G1(s)` is then the correct multiplicative reweight
turning that G1-shaped proposal into an unbiased estimator of a
G2(destination)-shaped target integral -- structurally the same role
`shift_maps.py`'s `solid_angle_ratio` Jacobian and
`heterogeneous_lookup.py`'s IS-reweight strategy play (a target-shape ratio
applied once), just for real 3D reconnection geometry instead of a
spectral shift.

Four estimators (self-normalized IS over `s`, matching this repo's own
`test_t16_...` self-normalization precedent for an unnormalized target):
  1. correct: weight = `J_reproj(s)` = `G2(s)/G1(s)`, spectral draw FRESH at
     the destination's own species-weight field `w1_dest(s)` -- unbiased.
  2. drop `J_reproj` (weight = 1) -- biased.
  3. correct weight but ALSO squared (`J_reproj(s)^2`, models wrongly
     applying the reconnection Jacobian twice / folding in a spurious
     extra factor) -- biased.
  4. correct weight, but STALE spectral draw from the generation-time
     species-weight field `w1_gen(s)` instead of the destination's
     `w1_dest(s)` -- biased.

**Cancellation-trap check done up front (T26's own lesson, see that file's
docstring):** `w1_gen`/`w1_dest` and the `x1`/`x2` embeddings are
deliberately asymmetric (different curves, different weight-field slopes)
-- confirmed via quadrature that all three broken estimators' own targets
differ from `TRUTH` by a real margin (>0.5%) before trusting the MC z-test,
same discipline as T26.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

torch.set_default_dtype(torch.float64)

Y = torch.tensor([0.5, 0.5, 1.0])  # fixed point light
N1 = torch.tensor([0.0, 0.0, 1.0])
N2 = torch.tensor([0.0, 0.0, 1.0])

MU1, MU2, SIGMA = 480.0, 620.0, 15.0


def x1_pos(s):
    """Frame-A (generation-time) shading-point curve, parametrized by s in [0,1]."""
    return torch.stack(
        [s, 0.15 * torch.sin(3 * torch.pi * s), torch.zeros_like(s)], dim=-1
    )


def x2_pos(s):
    """Frame-B (destination) shading-point curve -- a genuinely different
    embedding (real shear/offset, not a rigid translate of x1_pos)."""
    return torch.stack(
        [
            s + 0.15 * torch.sin(2 * torch.pi * s),
            0.6 * torch.sin(3 * torch.pi * s) - 0.25 * s,
            0.35 * torch.ones_like(s) - 0.3 * s,
        ],
        dim=-1,
    )


def cos_theta_o_and_dist(x, n):
    d = Y - x
    dist = torch.norm(d, dim=-1)
    cos = (d * n).sum(-1) / dist
    return cos, dist


def geometric_term(x, n):
    cos, dist = cos_theta_o_and_dist(x, n)
    return torch.clamp(cos, min=0.0) / dist ** 2


def w1_gen(s):
    """Generation-time (x1) species-weight field."""
    return 0.1 + 0.8 * s


def w1_dest(s):
    """Destination (x2) species-weight field -- deliberately different
    slope/intercept from w1_gen, not just a shifted copy (avoids the
    cancellation trap, see module docstring)."""
    return torch.clamp(0.75 - 0.65 * s, 0.0, 1.0)


def mean_lambda(s, w1_fn):
    return MU2 - (MU2 - MU1) * w1_fn(s)


def sample_mixture(s, w1_fn, rng):
    n = s.shape[0]
    p1 = w1_fn(s)
    is_species1 = torch.rand(n, generator=rng) < p1
    mu = torch.where(is_species1, torch.tensor(MU1), torch.tensor(MU2))
    return mu + SIGMA * torch.randn(n, generator=rng)


# --- quadrature setup (dense grid, machine-precision-level ground truth) ---
_S_GRID = torch.linspace(0.0, 1.0, 400_001)
_COS1, _DIST1 = cos_theta_o_and_dist(x1_pos(_S_GRID), N1)
_COS2, _DIST2 = cos_theta_o_and_dist(x2_pos(_S_GRID), N2)
_G1 = geometric_term(x1_pos(_S_GRID), N1)
_G2 = geometric_term(x2_pos(_S_GRID), N2)
_J_REPROJ_GRID = (_COS2 / _COS1) * (_DIST1 ** 2 / _DIST2 ** 2)

_Z1 = torch.trapz(_G1, _S_GRID).item()
_Z2 = torch.trapz(_G2, _S_GRID).item()
_G1_MAX = _G1.max().item()

TRUTH = (torch.trapz(_G2 * mean_lambda(_S_GRID, w1_dest), _S_GRID) / _Z2).item()
GT_DROP_J = (torch.trapz(_G1 * mean_lambda(_S_GRID, w1_dest), _S_GRID) / _Z1).item()
GT_SPURIOUS_J = (
    torch.trapz(_G1 * (_G2 / _G1) ** 2 * mean_lambda(_S_GRID, w1_dest), _S_GRID)
    / torch.trapz(_G1 * (_G2 / _G1) ** 2, _S_GRID)
).item()
GT_STALE_LAMBDA = (torch.trapz(_G2 * mean_lambda(_S_GRID, w1_gen), _S_GRID) / _Z2).item()


def test_cos_theta_o_never_goes_negative_over_the_domain():
    # matches the historical session's own domain-mismatch lesson: verify
    # the geometry never grazes/goes behind either surface before trusting
    # a clamp-free formula anywhere.
    assert bool((_COS1 > 0).all())
    assert bool((_COS2 > 0).all())


def test_reconnection_jacobian_matches_geometric_term_ratio_exactly():
    # G(x1)*J_reproj == G(x2) via two independently-computed paths -- the
    # historical session's own "confirms correct implementation" check,
    # ~1e-16 tightness expected.
    lhs = _G1 * _J_REPROJ_GRID
    rel_err = ((lhs - _G2).abs() / _G2.abs()).max().item()
    assert rel_err < 1e-12


def test_broken_estimators_ground_truths_differ_from_truth():
    for gt in (GT_DROP_J, GT_SPURIOUS_J, GT_STALE_LAMBDA):
        assert abs(gt - TRUTH) / TRUTH > 0.005


def _rejection_sample_s(n, rng):
    out = torch.empty(0)
    while out.shape[0] < n:
        batch = max(n, 1024)
        cand = torch.rand(batch, generator=rng)
        g = geometric_term(x1_pos(cand), N1)
        accept_p = g / _G1_MAX
        u = torch.rand(batch, generator=rng)
        out = torch.cat([out, cand[u < accept_p]])
    return out[:n]


def _self_normalized_estimate(vals, weights):
    return (vals * weights).sum().item() / weights.sum().item()


def _z_via_batches(vals, weights, truth, n_batches=30):
    n = vals.shape[0]
    batch_size = n // n_batches
    ests = torch.tensor(
        [
            _self_normalized_estimate(
                vals[b * batch_size:(b + 1) * batch_size],
                weights[b * batch_size:(b + 1) * batch_size],
            )
            for b in range(n_batches)
        ]
    )
    est = _self_normalized_estimate(vals, weights)
    sem = ests.std().item() / (n_batches ** 0.5)
    return (est - truth) / sem


def _draw(n, rng):
    s = _rejection_sample_s(n, rng)
    cos1, dist1 = cos_theta_o_and_dist(x1_pos(s), N1)
    cos2, dist2 = cos_theta_o_and_dist(x2_pos(s), N2)
    j_reproj = (cos2 / cos1) * (dist1 ** 2 / dist2 ** 2)
    lam_fresh = sample_mixture(s, w1_dest, rng)
    lam_stale = sample_mixture(s, w1_gen, rng)
    return j_reproj, lam_fresh, lam_stale


def test_correct_reweight_by_real_jacobian_is_unbiased():
    rng = torch.Generator().manual_seed(2700)
    j_reproj, lam_fresh, _ = _draw(300_000, rng)
    assert abs(_z_via_batches(lam_fresh, j_reproj, TRUTH)) < 3.5


def test_dropping_real_jacobian_is_biased():
    rng = torch.Generator().manual_seed(2701)
    _, lam_fresh, _ = _draw(300_000, rng)
    weights = torch.ones_like(lam_fresh)
    assert abs(_z_via_batches(lam_fresh, weights, GT_DROP_J)) < 3.5
    assert abs(_z_via_batches(lam_fresh, weights, TRUTH)) > 20.0


def test_squaring_the_real_jacobian_is_biased():
    rng = torch.Generator().manual_seed(2702)
    j_reproj, lam_fresh, _ = _draw(300_000, rng)
    weights = j_reproj ** 2
    assert abs(_z_via_batches(lam_fresh, weights, GT_SPURIOUS_J)) < 3.5
    assert abs(_z_via_batches(lam_fresh, weights, TRUTH)) > 20.0


def test_stale_spectral_reuse_is_biased():
    rng = torch.Generator().manual_seed(2703)
    j_reproj, _, lam_stale = _draw(300_000, rng)
    assert abs(_z_via_batches(lam_stale, j_reproj, GT_STALE_LAMBDA)) < 3.5
    assert abs(_z_via_batches(lam_stale, j_reproj, TRUTH)) > 20.0


if __name__ == "__main__":
    test_cos_theta_o_never_goes_negative_over_the_domain()
    test_reconnection_jacobian_matches_geometric_term_ratio_exactly()
    test_broken_estimators_ground_truths_differ_from_truth()
    test_correct_reweight_by_real_jacobian_is_unbiased()
    test_dropping_real_jacobian_is_biased()
    test_squaring_the_real_jacobian_is_biased()
    test_stale_spectral_reuse_is_biased()
    print("all T27 tests passed")
