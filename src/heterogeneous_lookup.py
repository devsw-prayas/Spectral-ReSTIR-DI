"""Heterogeneous local-lookup — checklist module 6.

Reuse-consistency machinery for A6's proved trichotomy
(restir_running_notes.md §7): a candidate y=(j, lambda') generated at vertex
z_A, reused (scored) at vertex z_B != z_A, for a rank-k fluorescent species
mixture with per-species concentration field c_j(z).

Target at a fixed vertex z: `p_hat(y;z) = c_j(z)*e_j(lambda_s)*a_j(lambda')*L_e(lambda')`,
optionally carrying a joint z/lambda' transmittance factor (e.g.
`exp(-optical_depth(z,lambda'))`) for the genuinely volumetric case.
Proposal at z: `q_z(y) = w_j(z)*pdf_j(lambda')`, species weight
`w_j(z) prop c_j(z)*e_j(lambda_s)*integral_j(z)` where `integral_j(z)` is
whatever normalizing integral the caller's candidate-generation scheme
actually computes for species j at z.

**Naive's cancellation trap.** If `w_j(z)` happens to equal the target's own
true marginal `integral(p_hat(j,.;z))` exactly, naive reuse (case 1 below)
is *always* exactly unbiased, for any z_A, z_B -- direct substitution shows
`E_{y~qA}[naive(y)] = sum_j P_A(j) = 1`-weighted average of `W(z_B)`,
independent of the mismatch between A and B. Genuine naive bias requires
the candidate-generation weight to be a genuine simplification that does
*not* track the true target's marginal (e.g. a proposal built from
`integral(a_j*L_e)` alone while the true target also carries a
z-and-lambda'-coupled transmittance term the proposal never accounts for)
-- this is what T7's toy probe and the tests below use, not an oversight.

Three reuse strategies (A6):
1. naive       -- score = p_hat(y;z_B)/q_{z_B}(y). Biased whenever q_zA!=q_zB.
2. IS-reweight -- score = p_hat(y;z_B)/q_{z_A}(y). Unbiased iff
   supp(p_hat(.;z_B)) subset supp(q_{z_A}); the moment a species is dead at
   z_A (c_j(z_A)==0) but alive at z_B, this is a structural bias, not a
   variance problem -- the mechanism behind T10's -78.4% result.
3. fix-local   -- fresh resample from q_{z_B}, y from z_A discarded
   entirely. Trivially unbiased (A7's free-path firewall corollary,
   freepath_sampler.py's module docstring, relies on exactly this branch's
   proof using no property of z_A's distribution).

All three strategies and the concentration/excitation/absorption/emission
functions are caller-supplied closures (species_concentration_fn(j, z),
excitation_fn(j, lambda_s), absorption_fn(j, lambda'), emission_fn(lambda'))
-- same convention as `freepath_sampler.py`'s `sigma_t_fn`: this module owns
none of the spectral shapes, only the reuse-consistency arithmetic.

Depends on: freepath_sampler (module 5) for the z_A/z_B vertices reused
across (this module doesn't sample them, just scores reuse given them).
"""


def species_weight(species_concentration_fn, excitation_fn, absorption_emission_integral_fn, j, z, lambda_s):
    """Un-normalized proposal weight `w_j(z) prop c_j(z)*e_j(lambda_s)*integral_j(z)`.

    `absorption_emission_integral_fn(j, z)` is the caller's own normalizing
    integral for species j's candidate generation at z -- pass one that
    ignores `z` (returns a constant per species) for the common case where
    candidate generation doesn't track a per-vertex transmittance term the
    true target carries (see module docstring, "naive's cancellation trap").
    """
    return (
        species_concentration_fn(j, z)
        * excitation_fn(j, lambda_s)
        * absorption_emission_integral_fn(j, z)
    )


def local_target(species_concentration_fn, excitation_fn, absorption_fn, emission_fn, j, lam_prime, z, lambda_s, transmittance_fn=None):
    """Un-normalized reuse target `p_hat((j, lambda'); z)` at fixed vertex z (A6).

    `transmittance_fn(j, lambda_prime, z)`, if given, multiplies in a joint
    z/lambda' coupling term (e.g. `exp(-optical_depth(z, lambda_prime))`).
    Without one, `p_hat` factors as (function of z) times (function of
    lambda'), which is exactly the degenerate case where naive reuse is
    always unbiased regardless of z_A vs z_B (see module docstring).
    """
    value = (
        species_concentration_fn(j, z)
        * excitation_fn(j, lambda_s)
        * absorption_fn(j, lam_prime)
        * emission_fn(lam_prime)
    )
    if transmittance_fn is not None:
        value = value * transmittance_fn(j, lam_prime, z)
    return value


def naive_score(target_fn, proposal_pdf_fn, y, z_dest):
    """Naive reuse score `p_hat(y;z_dest)/q_{z_dest}(y)` (A6 case 1).

    Biased whenever `q_zA != q_zB` -- direct consequence of
    `E_{y~qA}[f_B(y)/q_B(y)] != integral(f_B)` unless `q_A == q_B`.
    """
    q = proposal_pdf_fn(y, z_dest)
    if q == 0.0:
        return 0.0
    return target_fn(y, z_dest) / q


def is_reweight_score(target_fn, proposal_pdf_fn, y, z_source, z_dest):
    """IS-reweight reuse score `p_hat(y;z_dest)/q_{z_source}(y)` (A6 case 2).

    Returns `None` (existence failure -- caller must drop the term, never
    zero-fill, same discipline as `shift_maps`/`mis_combine`) when
    `q_{z_source}(y) == 0`: this is exactly the structural-bias mechanism
    (species dead at the source, alive at the destination), not a
    high-variance one.
    """
    q = proposal_pdf_fn(y, z_source)
    if q == 0.0:
        return None
    return target_fn(y, z_dest) / q


def fix_local_score(target_fn, proposal_pdf_fn, y, z_dest):
    """Fix-local reuse score (A6 case 3): `y` is a fresh candidate drawn from
    `q_{z_dest}` itself, so this is arithmetically identical to
    `naive_score` evaluated at its own vertex -- the content of this branch
    is the *decision* to discard whatever candidate arrived from `z_source`
    and resample fresh, not different arithmetic. Trivially unbiased, zero
    dependence on `z_source` (A7's firewall corollary uses exactly this).
    """
    return naive_score(target_fn, proposal_pdf_fn, y, z_dest)


def has_support_violation(species_concentration_fn, j, z_source, z_dest, tol=0.0):
    """True iff species `j` is structurally excluded from IS-reweight reuse:
    dead at `z_source` (`c_j(z_source) <= tol`) but alive at `z_dest`
    (`c_j(z_dest) > tol`) -- A6 case 2's failure condition.
    """
    return (
        species_concentration_fn(j, z_source) <= tol
        and species_concentration_fn(j, z_dest) > tol
    )


def lookup_trichotomy_case(species_concentration_fn, y, z_source, z_dest, tol=0.0):
    """Classify which A6 outcome governs IS-reweight-reusing sample
    `y=(j, lambda')` from `z_source` at `z_dest`.

    Returns `"support_violation"` if IS-reweight is structurally biased for
    this species (reuse must fall back to fix-local), else
    `"reweight_safe"` (IS-reweight is a valid unbiased choice, though
    possibly high-variance if the source/destination profiles diverge).
    Naive is not classified here: per case 1, it's suspect by default
    whenever `q_zA != q_zB`, never a case worth a "safe" verdict.
    """
    j, _ = y
    if has_support_violation(species_concentration_fn, j, z_source, z_dest, tol):
        return "support_violation"
    return "reweight_safe"
