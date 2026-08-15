"""Point-probe for T26 (composition lemma, 1D toy reprojection).

Claim under test: for temporal reuse under moving/deforming geometry, the
total shift Jacobian factors as `J_total = J_reproj` (pure geometric
reprojection), with the spectral degree of freedom handled entirely outside
the Jacobian via detached (fresh) resampling at the destination -- never
folded in as a multiplicative spectral-Jacobian factor, and never skippable.
This composition lemma is NOT yet built into
`shift_maps.py`/`temporal_history.py` (both of which are
locked to `reproject`-is-identity, static-geometry v1 scope) -- like T1
(rank-k reconnection), this is a standalone toy reconstruction with fresh
concrete parameters, not literal replay of the historical session's exact
numbers (which are scratch-only, `/home/claude/composition_lemma_probe/`,
never committed to any repo).

Toy: 1D position domain `x in [0,1]` in both frame A (source/history) and
frame B (destination/current), related by a smooth bijective reprojection
`x_B = phi(x_A)` with `phi(0)=0`, `phi(1)=1`, `phi' > 0` everywhere (a
genuine, nontrivial deformation -- not close to identity, `phi'` ranges over
roughly `[0.21, 1.79]`). A 2-species spectral mixture with a POSITION-
DEPENDENT (not symmetric -- see below) species-weight field `w1(x)` gives a
closed-form linear `mean_lambda(x)`, so quadrature ground truth is exact.

The estimated quantity is `Truth = integral_0^1 mean_lambda(x) dx`, i.e. the
uniform-over-frame-B-domain expectation of the destination's own spectral
mean. Four estimators, all built by drawing `x_A ~ Uniform[0,1]` and
`x_B = phi(x_A)`:
  1. correct: multiply by `J_reproj = phi'(x_A)`, draw lambda FRESH at `x_B`
     (destination's own mixture) -- unbiased for Truth by the standard
     change-of-variables identity.
  2. drop `J_reproj` entirely (assume rigid, `J=1`) -- biased.
  3. correct `J_reproj` but ALSO apply a spurious extra spectral Jacobian
     factor (`phi'(x_A)^2` instead of `phi'(x_A)`) -- models wrongly folding
     a "spectral shift Jacobian" into the reprojection term, which the
     composition lemma says should never happen (spectral part is detached
     resampling, Jacobian-free) -- biased.
  4. correct `J_reproj` but reuse the STALE lambda drawn from frame A's own
     mixture at `x_A` instead of resampling fresh at `x_B` -- biased.

**Non-obvious pitfall hit while designing this probe, worth remembering:**
a first attempt used a symmetric `w1(x) = 0.5 + 0.4*cos(pi*x)` (integrates
to a mean-preserving 0 over [0,1]) paired with an odd-symmetric `phi`. Under
that setup, estimators 2 and 4 came back statistically indistinguishable
from the TRUE answer (rel err <0.1%, |z|<1) even though they are *not*
algebraically valid estimators of Truth -- their own quadrature-computed
expectations happened to coincide with Truth by symmetry cancellation, the
exact same "cancellation trap" class flagged in `heterogeneous_lookup.py`'s
docstring and hit while building T7-T11. Fixed by using a genuinely
asymmetric linear `w1(x) = 0.1 + 0.8*x` (no reflective symmetry with `phi`),
which makes estimators 2/3/4 each converge to a DIFFERENT, wrong number
(confirmed both algebraically via quadrature and empirically via MC).
Moral: a toy composition/reprojection probe needs an asymmetric deformation
or weight field, or "broken" estimators can pass by symmetry accident.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

torch.set_default_dtype(torch.float64)

MU1, MU2, SIGMA = 480.0, 620.0, 15.0
TRUTH = 550.0  # integral_0^1 mean_lambda(x) dx, exact closed form (see below)


def phi(x):
    """Nontrivial monotonic reprojection [0,1] -> [0,1], phi(0)=0, phi(1)=1."""
    return x + 0.25 * torch.sin(torch.pi * x)


def phi_prime(x):
    """J_reproj = dx_B/dx_A."""
    return 1.0 + 0.25 * torch.pi * torch.cos(torch.pi * x)


def w1(x):
    """Species-1 weight field, deliberately asymmetric (see module docstring)."""
    return 0.1 + 0.8 * x


def mean_lambda(x):
    """E[lambda | x] under the 2-species mixture -- linear, closed form."""
    return MU2 - (MU2 - MU1) * w1(x)


def sample_mixture(x, rng):
    n = x.shape[0]
    p1 = w1(x)
    is_species1 = torch.rand(n, generator=rng) < p1
    mu = torch.where(is_species1, torch.tensor(MU1), torch.tensor(MU2))
    return mu + SIGMA * torch.randn(n, generator=rng)


def _quadrature_truth(fn, n=400_001):
    x = torch.linspace(0.0, 1.0, n)
    return torch.trapz(fn(x), x).item()


# ground truths for the three BROKEN estimators' own (wrong) target
# quantities, via quadrature -- confirms each estimator is internally
# consistent with its own algebra, distinct from and biased relative to
# TRUTH (checked by the MC tests below).
GT_DROP_J = _quadrature_truth(lambda x1: mean_lambda(phi(x1)))
GT_SPURIOUS_J = _quadrature_truth(lambda x1: mean_lambda(phi(x1)) * phi_prime(x1) ** 2)
GT_STALE_LAMBDA = _quadrature_truth(lambda x1: mean_lambda(x1) * phi_prime(x1))


def test_truth_is_exact_closed_form():
    # integral_0^1 mean_lambda(x) dx = MU2 - (MU2-MU1)*integral_0^1 w1(x) dx
    # = 620 - 140*(0.1 + 0.8*0.5) = 620 - 140*0.5 = 550, exact.
    expected = MU2 - (MU2 - MU1) * _quadrature_truth(w1)
    assert abs(expected - TRUTH) < 1e-9


def test_phi_is_a_valid_bijection_on_unit_interval():
    grid = torch.linspace(0.0, 1.0, 200_001)
    assert abs(phi(torch.tensor(0.0)).item()) < 1e-12
    assert abs(phi(torch.tensor(1.0)).item() - 1.0) < 1e-12
    assert bool((phi_prime(grid) > 0).all())
    mapped = phi(grid)
    assert bool((mapped[1:] > mapped[:-1]).all())


def test_broken_estimators_ground_truths_differ_from_truth():
    # sanity: the three broken estimators' own quadrature-exact targets are
    # each measurably different from TRUTH=550 -- if any of these collapsed
    # back to ~550 the toy would be degenerate (the cancellation trap this
    # module's docstring warns about).
    for gt in (GT_DROP_J, GT_SPURIOUS_J, GT_STALE_LAMBDA):
        assert abs(gt - TRUTH) / TRUTH > 0.02


def _draw(n, rng):
    x_a = torch.rand(n, generator=rng)
    x_b = phi(x_a)
    j_reproj = phi_prime(x_a)
    lam_fresh_at_b = sample_mixture(x_b, rng)
    lam_stale_at_a = sample_mixture(x_a, rng)
    return x_a, x_b, j_reproj, lam_fresh_at_b, lam_stale_at_a


def _z(samples, truth):
    n = samples.shape[0]
    mean = samples.mean().item()
    stderr = samples.std().item() / (n ** 0.5)
    return (mean - truth) / stderr


def test_correct_composition_is_unbiased_for_truth():
    rng = torch.Generator().manual_seed(2600)
    _, _, j_reproj, lam_fresh, _ = _draw(500_000, rng)
    samples = lam_fresh * j_reproj
    assert abs(_z(samples, TRUTH)) < 3.5


def test_dropping_j_reproj_is_biased():
    rng = torch.Generator().manual_seed(2601)
    _, _, _, lam_fresh, _ = _draw(500_000, rng)
    samples = lam_fresh  # J_reproj dropped entirely
    assert abs(_z(samples, GT_DROP_J)) < 3.5  # matches its own (wrong) target
    assert abs(_z(samples, TRUTH)) > 20.0  # decisively biased for the real Truth


def test_spurious_extra_spectral_jacobian_is_biased():
    rng = torch.Generator().manual_seed(2602)
    _, _, j_reproj, lam_fresh, _ = _draw(500_000, rng)
    samples = lam_fresh * j_reproj ** 2  # bogus extra factor
    assert abs(_z(samples, GT_SPURIOUS_J)) < 3.5
    assert abs(_z(samples, TRUTH)) > 20.0


def test_stale_lambda_reuse_is_biased():
    rng = torch.Generator().manual_seed(2603)
    _, _, j_reproj, _, lam_stale = _draw(500_000, rng)
    samples = lam_stale * j_reproj  # correct J_reproj, but stale spectral sample
    assert abs(_z(samples, GT_STALE_LAMBDA)) < 3.5
    assert abs(_z(samples, TRUTH)) > 20.0


if __name__ == "__main__":
    test_truth_is_exact_closed_form()
    test_phi_is_a_valid_bijection_on_unit_interval()
    test_broken_estimators_ground_truths_differ_from_truth()
    test_correct_composition_is_unbiased_for_truth()
    test_dropping_j_reproj_is_biased()
    test_spurious_extra_spectral_jacobian_is_biased()
    test_stale_lambda_reuse_is_biased()
    print("all T26 tests passed")
