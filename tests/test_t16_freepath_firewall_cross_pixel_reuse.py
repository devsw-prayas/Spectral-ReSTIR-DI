"""Point-probe for T16 (session_log_restir_4 V-E-2: free-path firewall,
cross-pixel reuse under real, independently free-path-sampled z_A and z_B).

T9's counter-gradient rank-2 trichotomy (naive biased / IS-reweight
unbiased-but-costly / fix-local unbiased-and-cheap) used EXTERNALLY FIXED
Z_A=0.2, Z_B=0.8. This probe re-runs the identical target/proposal/species
machinery, but z_A and z_B are now genuine random variables drawn each trial
via `freepath_sampler.sample_free_path` (module 5) -- the actual content of
A7's firewall corollary and the one thing T2-T14 never tested (all fixed z).
Position sampling uses its OWN wavelength context per pixel
(`POS_LAMBDA_A=560`, `POS_LAMBDA_B=620`, mirroring the two species' peaks so
z_A/z_B distributions genuinely diverge), kept structurally separate from
the species/wavelength candidate-generation machinery's own fixed
`LAMBDA_S=590` context -- this separation IS the "position-sampling
firewall" (freepath_sampler.py module docstring, A2): the two wavelength
roles never leak into each other.

Because z_B is now random, the ground truth is the DOUBLE integral
`integral_0^1 integral target_fn(y,z) dy dz` (marginalizing over z_B too),
recovered by dividing each trial's score by the sampler's own hit density
`p(z_B)` (same "z_B is the outer GT(B) integration variable... standard
importance weight 1/p(z_B)" convention the session log states explicitly).
z_A is treated as data (conditioned on landing -- resampled until a real
collision, since it plays no denominator role, matching the session log's
"z_A ... does not get its own importance weight, since nothing here
separately estimates GT(A)").

fix-local's estimator doesn't depend on z_A at all, so its unbiasedness for
GT is unconditional. naive/IS-reweight's conditional-on-(z_A,z_B) bias
mechanism is unchanged from T9 (same target/proposal mismatch); this probe
checks that mechanism survives being averaged over genuinely random,
independently-sampled z_A, z_B rather than one fixed adversarial pair.
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from heterogeneous_lookup import (
    species_weight,
    local_target,
    naive_score,
    is_reweight_score,
    fix_local_score,
)
from freepath_sampler import sample_free_path

torch.set_default_dtype(torch.float64)

LAMBDA = torch.linspace(400.0, 700.0, 1500)
DLAM = (LAMBDA[1] - LAMBDA[0]).item()
LAMBDA_S = 590.0  # candidate-generation excitation context, fixed -- unrelated to position sampling
MU = {0: 560.0, 1: 620.0}
SIGMA = {0: 40.0, 1: 40.0}
EMISSION_MU, EMISSION_SIGMA = 590.0, 70.0
Z0, WIDTH = 0.5, 0.05  # sigmoid transition center/width, same as T9
KP = 1.0 / WIDTH
L_SCALE = 300.0  # self-absorption strength, same as T9

POS_LAMBDA_A, POS_LAMBDA_B = 560.0, 620.0  # position-sampling wavelength context, per pixel
POS_SIGMA = 40.0
FLOOR = 0.05  # small constant background extinction (matches session log's sigma_bg)


def _gaussian_pdf(x, mu, sigma):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


A_TENSOR = {j: _gaussian_pdf(LAMBDA, MU[j], SIGMA[j]) for j in (0, 1)}
LE_TENSOR = _gaussian_pdf(LAMBDA, EMISSION_MU, EMISSION_SIGMA)
K_NO_TRANS = {j: torch.trapz(A_TENSOR[j] * LE_TENSOR, LAMBDA).item() for j in (0, 1)}


def absorption(j, lam):
    return _gaussian_pdf(torch.as_tensor(float(lam)), MU[j], SIGMA[j]).item()


def emission(lam):
    return _gaussian_pdf(torch.as_tensor(float(lam)), EMISSION_MU, EMISSION_SIGMA).item()


def excitation(j, lambda_s):
    return 1.0


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def conc_counter(j, z):
    s = _sigmoid((z - Z0) / WIDTH)
    return 2.0 - 1.9 * s if j == 0 else 0.1 + 1.9 * s


_Z_GRID = torch.linspace(0.0, 1.0, 2001)
_DZ = (_Z_GRID[1] - _Z_GRID[0]).item()


def _build_column_density(j):
    vals = torch.tensor([conc_counter(j, z.item()) for z in _Z_GRID])
    trailing = torch.zeros_like(vals)
    total = 0.0
    for i in range(len(vals) - 2, -1, -1):
        total += 0.5 * (vals[i].item() + vals[i + 1].item()) * _DZ
        trailing[i] = total
    return trailing


_COLDENS = {j: _build_column_density(j) for j in (0, 1)}


def _column_density(j, z):
    idx = torch.searchsorted(_Z_GRID, torch.as_tensor(float(z))).clamp(max=len(_Z_GRID) - 1)
    return _COLDENS[j][idx].item()


def transmittance_counter(j, lam_prime, z):
    optical_depth = L_SCALE * sum(
        absorption(k, lam_prime) * _column_density(k, z) for k in (0, 1)
    )
    return math.exp(-optical_depth)


def _no_trans_integral(j, z):
    return K_NO_TRANS[j]


def target_fn(y, z):
    j, lam = y
    return local_target(
        conc_counter, excitation, absorption, emission, j, lam, z, LAMBDA_S,
        transmittance_fn=transmittance_counter,
    )


def proposal_pdf_fn(y, z):
    j, lam = y
    weights = {k: species_weight(conc_counter, excitation, _no_trans_integral, k, z, LAMBDA_S) for k in (0, 1)}
    total = weights[0] + weights[1]
    if total == 0.0:
        return 0.0
    return (weights[j] / total) * absorption(j, lam)


def _mix(z):
    weights = {k: species_weight(conc_counter, excitation, _no_trans_integral, k, z, LAMBDA_S) for k in (0, 1)}
    total = weights[0] + weights[1]
    return weights[0] / total, weights[1] / total


def _sample(z, rng):
    w0, _ = _mix(z)
    j = 0 if torch.rand((), generator=rng).item() < w0 else 1
    lam = torch.normal(MU[j], SIGMA[j], (1,), generator=rng).item()
    return j, lam


# --- position-sampling machinery (module 5), decoupled from the above ---

def _excitation_pos(j, lambda_s):
    return math.exp(-0.5 * ((lambda_s - MU[j]) / POS_SIGMA) ** 2)


def sigma_t(z, lambda_s):
    return conc_counter(0, z) * _excitation_pos(0, lambda_s) + conc_counter(1, z) * _excitation_pos(1, lambda_s) + FLOOR


def _softplus(x):
    return math.log1p(math.exp(x)) if x < 30 else x


def _int_conc0(z):
    return 2.0 * z - (1.9 / KP) * (_softplus(KP * (z - Z0)) - _softplus(-KP * Z0))


def _int_conc1(z):
    return 0.1 * z + (1.9 / KP) * (_softplus(KP * (z - Z0)) - _softplus(-KP * Z0))


def optical_depth_pos(z, lambda_s):
    return _excitation_pos(0, lambda_s) * _int_conc0(z) + _excitation_pos(1, lambda_s) * _int_conc1(z) + FLOOR * z


def pdf_pos(z, lambda_s):
    return sigma_t(z, lambda_s) * math.exp(-optical_depth_pos(z, lambda_s))


def _majorant(lambda_s, n=300):
    return max(sigma_t(z / (n - 1), lambda_s) for z in range(n)) * 1.02


MAJORANT_A = _majorant(POS_LAMBDA_A)
MAJORANT_B = _majorant(POS_LAMBDA_B)


def _sigma_t_a(z):
    return sigma_t(z, POS_LAMBDA_A)


def _sigma_t_b(z):
    return sigma_t(z, POS_LAMBDA_B)


def _sample_hit(sigma_t_fn_1arg, majorant, rng):
    while True:
        z, is_hit = sample_free_path(sigma_t_fn_1arg, majorant, 0.0, 1.0, rng)
        if is_hit:
            return z


# --- ground truth: double integral over z_B and (species, lambda') ---

def _inner_integral(z):
    total = 0.0
    for j in (0, 1):
        for lam in LAMBDA:
            total += target_fn((j, lam.item()), z) * DLAM
    return total


def _ground_truth(n_z=60):
    zs = torch.linspace(0.0, 1.0, n_z)
    vals = torch.tensor([_inner_integral(z.item()) for z in zs])
    return torch.trapz(vals, zs).item()


GT = _ground_truth()


def test_position_sampling_context_pulls_a_and_b_apart():
    """Sanity: A's position context (560, species 0's peak) should draw z_A
    preferentially toward species 0's high-concentration side (low z), and
    B's context (620) toward species 1's side (high z) -- the genuinely
    lambda_s-coupled free-path claim this whole probe rests on."""
    rng = torch.Generator().manual_seed(1600)
    N = 3000
    z_a_samples = torch.tensor([_sample_hit(_sigma_t_a, MAJORANT_A, rng) for _ in range(N)])
    z_b_samples = torch.tensor([_sample_hit(_sigma_t_b, MAJORANT_B, rng) for _ in range(N)])
    assert z_a_samples.mean().item() < z_b_samples.mean().item() - 0.1


def _run(N, seed):
    rng = torch.Generator().manual_seed(seed)
    naive_samples = torch.empty(N)
    isr_samples = torch.empty(N)
    fixl_samples = torch.empty(N)

    for t in range(N):
        z_a = _sample_hit(_sigma_t_a, MAJORANT_A, rng)
        z_b, hit_b = sample_free_path(_sigma_t_b, MAJORANT_B, 0.0, 1.0, rng)

        if not hit_b:
            naive_samples[t] = 0.0
            isr_samples[t] = 0.0
            fixl_samples[t] = 0.0
            continue

        p_zb = pdf_pos(z_b, POS_LAMBDA_B)

        y_from_a = _sample(z_a, rng)
        naive_val = naive_score(target_fn, proposal_pdf_fn, y_from_a, z_b)
        isr_val = is_reweight_score(target_fn, proposal_pdf_fn, y_from_a, z_a, z_b)

        y_from_b = _sample(z_b, rng)
        fixl_val = fix_local_score(target_fn, proposal_pdf_fn, y_from_b, z_b)

        naive_samples[t] = naive_val / p_zb
        isr_samples[t] = (0.0 if isr_val is None else isr_val) / p_zb
        fixl_samples[t] = fixl_val / p_zb

    return naive_samples, isr_samples, fixl_samples


def test_naive_decisively_biased_under_real_random_z():
    N = 60_000
    naive, _, _ = _run(N, seed=21)
    mean = naive.mean().item()
    stderr = naive.std().item() / (N ** 0.5)
    z_score = (mean - GT) / stderr
    assert abs(z_score) > 20.0
    assert abs((mean - GT) / GT) > 0.1


def test_fix_local_unbiased_regardless_of_za():
    N = 60_000
    _, _, fixl = _run(N, seed=21)
    mean = fixl.mean().item()
    assert abs((mean - GT) / GT) < 0.05


def test_is_reweight_unbiased_but_far_higher_variance_than_fix_local():
    N = 60_000
    _, isr, fixl = _run(N, seed=21)

    mean_isr = isr.mean().item()
    assert abs((mean_isr - GT) / GT) < 0.15  # unbiased-ish (heavy-tailed variance, not systematic)

    variance_ratio = isr.var().item() / fixl.var().item()
    assert variance_ratio > 5.0  # same "correctness survives, efficiency doesn't" shape as T9


if __name__ == "__main__":
    test_position_sampling_context_pulls_a_and_b_apart()
    test_naive_decisively_biased_under_real_random_z()
    test_fix_local_unbiased_regardless_of_za()
    test_is_reweight_unbiased_but_far_higher_variance_than_fix_local()
    print("all T16 tests passed")
