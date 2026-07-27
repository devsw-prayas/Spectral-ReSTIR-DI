"""Point-probe for T1 (Phase 1.1 rank-k surface reconnection probe).

Historical source: `.claude/ref/addendum_fable5_crosscheck_phase1_1_findings.md`,
Part B item 4 -- the ONLY surviving record (the original scratch scene at
`/home/claude/rankk_probe/` is lost). Only the qualitative outcome survives:
naive reuse ~70.8% biased, IS-reweight suffers severe variance blowup
(~96% relative std at N=2e6) for spectrally well-separated species, detached
joint-resample (Zeltner et al. 2021's category) confirmed unbiased. This
file RECONSTRUCTS the mechanism with fresh concrete numbers -- not a literal
replay of the lost scene -- same discipline as G8's retirement note
(`forward_paper1_test_suite.md`) about not treating an unrecoverable
historical number as gospel.

Rank-k (k=2 species here) generalizes the rank-1 recorrelation lemma
(`recorrelation_sampler.py`): species-mixing weights `pi_j(lambda_in)`
genuinely DEPEND on the incoming-wavelength context, unlike the rank-1 case
where there's only one species (mixing is trivial). No formal rank-k
reconnection-validity theorem exists (flagged gap, T1/T7-T10 numeric-only
per `forward_paper1_test_suite.md`) -- this is a standalone probe, not built
on `shift_maps.py`/`mis_combine.py` (those are locked to the rank-1 scope).

Mechanism: a candidate (species j, wavelength lambda) is generated at a
SOURCE vertex under the source's own species-mixing weights `pi^S`. Reusing
it at a DESTINATION vertex whose mixing weights `pi^D` differ requires
either (a) reweighting by the species-only importance ratio
`pi^D[j]/pi^S[j]` (the emission PMF `e_j(lambda)` cancels identically --
IS-reweight), or (b) discarding the stale candidate and jointly resampling
(species, lambda) fresh from `pi^D` (detached-fix). Naively treating the
source-generated candidate as already representative of the destination
(no reweight at all) is the bug.
"""

import torch

torch.set_default_dtype(torch.float64)

LAMBDAS = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
E0 = torch.tensor([0.50, 0.30, 0.15, 0.05, 0.00])  # species 0 re-emission PMF
E1 = torch.tensor([0.00, 0.05, 0.15, 0.30, 0.50])  # species 1 re-emission PMF, moderate overlap w/ E0

PI_SRC = torch.tensor([0.98, 0.02])  # source context: species 1 rare
PI_DST = torch.tensor([0.20, 0.80])  # destination context: species 1 dominant -- spectrally well-separated regime

TRUTH = (PI_DST[0] * (LAMBDAS * E0).sum() + PI_DST[1] * (LAMBDAS * E1).sum()).item()


def _draw_species_and_lambda(pi, n, rng):
    """Vectorized draw of (species, lambda) ~ pi[j] * e_j(lambda)."""
    js = torch.multinomial(pi, n, replacement=True, generator=rng)
    lam_idx = torch.empty(n, dtype=torch.long)
    mask0 = js == 0
    n0 = int(mask0.sum().item())
    n1 = n - n0
    if n0 > 0:
        lam_idx[mask0] = torch.multinomial(E0, n0, replacement=True, generator=rng)
    if n1 > 0:
        lam_idx[~mask0] = torch.multinomial(E1, n1, replacement=True, generator=rng)
    return js, LAMBDAS[lam_idx]


def test_truth_is_a_genuine_mixture_reflecting_destination_context():
    assert abs(TRUTH - 2.75) < 1e-9  # closed-form check on the constructed scene


def test_naive_reuse_is_silently_biased():
    # Core T1 finding: reusing the source-generated (species, lambda)
    # candidate at the destination with NO reweighting is not a variance
    # problem -- it's a clean, deterministic bias equal to E^src[lambda]
    # instead of the true E^dst[lambda].
    N = 200_000
    rng = torch.Generator().manual_seed(1)
    _, lams = _draw_species_and_lambda(PI_SRC, N, rng)
    naive_mean = lams.mean().item()

    rel_bias = (naive_mean - TRUTH) / TRUTH
    assert abs(rel_bias) > 0.5  # large, not a rounding error (reconstruction lands near historical ~70.8%)


def test_is_reweight_unbiased_but_variance_blows_up():
    # Correct importance-sampling reweight: w = pi_dst[j]/pi_src[j] (the
    # e_j(lambda) factor cancels identically -- a species-only ratio).
    # Exact in expectation, but species 1 is rare at the source (2%) yet
    # dominant at the destination (80%) -- the spectrally-well-separated
    # case -- so its rare draws carry a 40x weight: the classic
    # heavy-tailed IS blowup.
    N = 2_000_000
    rng = torch.Generator().manual_seed(2)
    js, lams = _draw_species_and_lambda(PI_SRC, N, rng)
    weight_ratio = (PI_DST / PI_SRC)[js]
    samples = lams * weight_ratio

    mean = samples.mean().item()
    std = samples.std().item()

    assert abs(mean - TRUTH) / TRUTH < 0.02  # unbiased
    assert std / mean > 1.0  # severe variance blowup (>100% relative per-sample std)


def test_detached_fix_is_unbiased_and_low_variance():
    # Zeltner et al. detached-resampling category: at reconnection time,
    # jointly resample (species, lambda) fresh from the destination target
    # instead of reweighting the stale source-generated candidate.
    N = 200_000
    rng = torch.Generator().manual_seed(3)
    _, lams = _draw_species_and_lambda(PI_DST, N, rng)

    mean = lams.mean().item()
    stderr = lams.std().item() / (N ** 0.5)
    z = (mean - TRUTH) / stderr

    assert abs(z) < 3.5
    assert lams.std().item() / mean < 1.0  # comfortably away from the IS-reweight blowup regime


if __name__ == "__main__":
    test_truth_is_a_genuine_mixture_reflecting_destination_context()
    test_naive_reuse_is_silently_biased()
    test_is_reweight_unbiased_but_variance_blows_up()
    test_detached_fix_is_unbiased_and_low_variance()
    print("all T1 tests passed")
