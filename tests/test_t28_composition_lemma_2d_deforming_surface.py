"""Point-probe for T28 (composition lemma on a genuine 2D deforming
surface, real area Jacobian).

T26 used an abstract 1D reprojection map; T27 used the real single-vertex
point-light reconnection Jacobian but stayed 1D. T28 is the third and
final composition-lemma leg: a genuine 2D surface parametrized by
`(u,v) in [0,1]^2`, given TWO independent, nontrivial embeddings into R^3
-- frame A (`XA`, a mild Gaussian bump near the origin corner) and frame B
(`XB`, a larger bump near the opposite corner PLUS real in-plane shear,
`u`/`v` each perturbed by a sine of the OTHER coordinate) -- not a rigid
translate of A. The composition-lemma quantity under test here is the pure
AREA Jacobian ratio (no point light / cos-theta term, unlike T27):

    J_total = J_area,B(u,v) / J_area,A(u,v)

`J_area` is computed via CENTRAL FINITE DIFFERENCES on the embedding
(`(X(u+h,v)-X(u-h,v))/2h` cross `(X(u,v+h)-X(u,v-h))/2h`, magnitude),
matching the historical session's own methodology choice ("not
hand-derived, to avoid calculus errors") rather than a symbolically
differentiated closed form.

Candidates `(u,v)` are drawn via rejection sampling proportional to
`J_area,A(u,v)` (genuinely simulating "historically area-uniform
resampling on frame-A's own geometry" -- same technique as T27's `G1`-
proportional rejection sampling and the historical T1.3's own method).
Same 4-way battery as T26/T27:
  1. correct: weight = `J_total(u,v)`, spectral draw FRESH at destination's
     `w1_dest(u,v)` field -- unbiased.
  2. drop the ratio (assume rigid, weight=1) -- biased.
  3. correct ratio, but SQUARED (spurious extra factor) -- biased.
  4. correct ratio, but STALE spectral draw from `w1_gen(u,v)` -- biased.

**Design note, matching T26/T27's own documented cancellation-trap
lesson:** a first attempt used near-flat, mild bump embeddings for both A
and B (`J_area,A` in `[1.0,1.11]`, i.e. nearly uniform) -- estimators 2/3
came back with sub-0.1%-relative bias, too thin a margin for a reliable MC
threshold. Fixed by making the two embeddings' bumps peak at OPPOSITE
corners with substantially different magnitude/spread, confirmed via
quadrature that all three broken estimators diverge from `TRUTH` by
several percent before trusting the MC battery.

**Historical caveat this test does NOT need to reproduce:** the original
session reported estimator 3 (spurious extra factor) as NOT detectably
biased in that specific run (a reported limitation of statistical
bias-detection for that particular deformation, not evidence the broken
form is unbiased -- see that section's own honest write-up). This toy's
parameters give estimator 3 a clear, detectable bias instead (confirmed by
quadrature before writing the MC assertion) -- a different, but equally
legitimate, concrete instance of the same general claim.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

torch.set_default_dtype(torch.float64)

MU1, MU2, SIGMA = 480.0, 620.0, 15.0
_FD_H = 1e-5


def XA(uv):
    """Frame-A embedding: mild bump near (0.2, 0.2)."""
    u, v = uv[..., 0], uv[..., 1]
    bump = 0.7 * torch.exp(-((u - 0.2) ** 2 + (v - 0.2) ** 2) / (2 * 0.12 ** 2))
    return torch.stack([u, v, bump], dim=-1)


def XB(uv):
    """Frame-B embedding: larger bump near (0.8, 0.8) plus real in-plane shear."""
    u, v = uv[..., 0], uv[..., 1]
    bump = 1.3 * torch.exp(-((u - 0.8) ** 2 + (v - 0.8) ** 2) / (2 * 0.18 ** 2))
    return torch.stack(
        [u + 0.25 * torch.sin(torch.pi * v), v + 0.25 * torch.sin(torch.pi * u), bump],
        dim=-1,
    )


def area_jacobian(embed_fn, uv):
    """|dX/du x dX/dv| via central finite differences (not hand-derived)."""
    u, v = uv[..., 0], uv[..., 1]
    x_u_plus = embed_fn(torch.stack([u + _FD_H, v], dim=-1))
    x_u_minus = embed_fn(torch.stack([u - _FD_H, v], dim=-1))
    x_v_plus = embed_fn(torch.stack([u, v + _FD_H], dim=-1))
    x_v_minus = embed_fn(torch.stack([u, v - _FD_H], dim=-1))
    x_u = (x_u_plus - x_u_minus) / (2 * _FD_H)
    x_v = (x_v_plus - x_v_minus) / (2 * _FD_H)
    return torch.norm(torch.cross(x_u, x_v, dim=-1), dim=-1)


def w1_gen(u, v):
    return torch.clamp(0.1 + 0.5 * u + 0.3 * v, 0.0, 1.0)


def w1_dest(u, v):
    return torch.clamp(0.8 - 0.5 * u - 0.2 * v, 0.0, 1.0)


def mean_lambda(u, v, w1_fn):
    return MU2 - (MU2 - MU1) * w1_fn(u, v)


def sample_mixture(u, v, w1_fn, rng):
    n = u.shape[0]
    p1 = w1_fn(u, v)
    is_species1 = torch.rand(n, generator=rng) < p1
    mu = torch.where(is_species1, torch.tensor(MU1), torch.tensor(MU2))
    return mu + SIGMA * torch.randn(n, generator=rng)


def _trapz2d(f, u_grid, v_grid):
    return torch.trapz(torch.trapz(f, v_grid, dim=1), u_grid, dim=0)


_U_GRID = torch.linspace(0.0, 1.0, 401)
_V_GRID = torch.linspace(0.0, 1.0, 401)
_UU, _VV = torch.meshgrid(_U_GRID, _V_GRID, indexing="ij")
_UV_GRID = torch.stack([_UU, _VV], dim=-1)
_JA = area_jacobian(XA, _UV_GRID)
_JB = area_jacobian(XB, _UV_GRID)

_ZA = _trapz2d(_JA, _U_GRID, _V_GRID).item()
_ZB = _trapz2d(_JB, _U_GRID, _V_GRID).item()
_JA_MAX = _JA.max().item()

_MLAM_DEST = mean_lambda(_UU, _VV, w1_dest)
_MLAM_GEN = mean_lambda(_UU, _VV, w1_gen)

TRUTH = (_trapz2d(_JB * _MLAM_DEST, _U_GRID, _V_GRID) / _ZB).item()
GT_DROP_J = (_trapz2d(_JA * _MLAM_DEST, _U_GRID, _V_GRID) / _ZA).item()
_RATIO_SQ = (_JB / _JA) ** 2
GT_SPURIOUS_J = (
    _trapz2d(_JA * _RATIO_SQ * _MLAM_DEST, _U_GRID, _V_GRID)
    / _trapz2d(_JA * _RATIO_SQ, _U_GRID, _V_GRID)
).item()
GT_STALE_LAMBDA = (_trapz2d(_JB * _MLAM_GEN, _U_GRID, _V_GRID) / _ZB).item()


def test_area_jacobians_are_positive_everywhere():
    assert bool((_JA > 0).all())
    assert bool((_JB > 0).all())


def test_broken_estimators_ground_truths_differ_from_truth():
    for gt in (GT_DROP_J, GT_SPURIOUS_J, GT_STALE_LAMBDA):
        assert abs(gt - TRUTH) / TRUTH > 0.01


def _rejection_sample_uv(n, rng):
    out = torch.empty(0, 2)
    while out.shape[0] < n:
        batch = max(n, 4096)
        cand = torch.rand(batch, 2, generator=rng)
        g = area_jacobian(XA, cand)
        accept_p = g / _JA_MAX
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
    uv = _rejection_sample_uv(n, rng)
    u, v = uv[:, 0], uv[:, 1]
    j_total = area_jacobian(XB, uv) / area_jacobian(XA, uv)
    lam_fresh = sample_mixture(u, v, w1_dest, rng)
    lam_stale = sample_mixture(u, v, w1_gen, rng)
    return j_total, lam_fresh, lam_stale


def test_correct_area_jacobian_ratio_is_unbiased():
    rng = torch.Generator().manual_seed(2800)
    j_total, lam_fresh, _ = _draw(300_000, rng)
    assert abs(_z_via_batches(lam_fresh, j_total, TRUTH)) < 3.5


def test_dropping_area_jacobian_ratio_is_biased():
    rng = torch.Generator().manual_seed(2801)
    _, lam_fresh, _ = _draw(300_000, rng)
    weights = torch.ones_like(lam_fresh)
    assert abs(_z_via_batches(lam_fresh, weights, GT_DROP_J)) < 3.5
    assert abs(_z_via_batches(lam_fresh, weights, TRUTH)) > 20.0


def test_squaring_area_jacobian_ratio_is_biased():
    rng = torch.Generator().manual_seed(2802)
    j_total, lam_fresh, _ = _draw(300_000, rng)
    weights = j_total ** 2
    assert abs(_z_via_batches(lam_fresh, weights, GT_SPURIOUS_J)) < 3.5
    assert abs(_z_via_batches(lam_fresh, weights, TRUTH)) > 20.0


def test_stale_spectral_reuse_is_biased():
    rng = torch.Generator().manual_seed(2803)
    j_total, _, lam_stale = _draw(300_000, rng)
    assert abs(_z_via_batches(lam_stale, j_total, GT_STALE_LAMBDA)) < 3.5
    assert abs(_z_via_batches(lam_stale, j_total, TRUTH)) > 20.0


if __name__ == "__main__":
    test_area_jacobians_are_positive_everywhere()
    test_broken_estimators_ground_truths_differ_from_truth()
    test_correct_area_jacobian_ratio_is_unbiased()
    test_dropping_area_jacobian_ratio_is_biased()
    test_squaring_area_jacobian_ratio_is_biased()
    test_stale_spectral_reuse_is_biased()
    print("all T28 tests passed")
