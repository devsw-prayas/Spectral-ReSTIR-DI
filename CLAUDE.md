# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Infrastructure for "Spectral ReSTIR DI" (Forward Paper 1), companion to
"Inverse Spectral Rendering" (Inverse Paper 1). It is not a renderer — it's a
`torch`-based library of small, independently-testable numerical modules that
the paper's toy point-probes (T-tier tests, see below) are built on top of,
plus the theory notes those modules implement.

## Commands

```
python src/check_env.py            # verify Spectral conda env: torch/scipy/matplotlib, CUDA, float64 default
python tests/test_<module>.py      # run one module's tests directly (no pytest — plain asserts + a __main__ block)
```

There is no test runner, build step, or lint config — each `tests/test_*.py`
is a standalone script with `assert`-based probes and an
`if __name__ == "__main__":` block that runs them all and prints
`"all <module> tests passed"`. Run the file directly; there is no pytest
dependency in this repo.

Use `C:/Users/Dell/anaconda3/envs/Spectral/python.exe`, not a bare `python`,
if the conda env isn't already activated — the base interpreter has no torch.

## Architecture

**Everything is float64.** `torch.set_default_dtype(torch.float64)` must be
called at the entry point of any script/test before using these modules
(see the top of any `tests/test_*.py` or `check_env.py`'s `__main__`).

**Tier 0** (`snell_jacobian.py`, `cauchy_ior.py`, `spectral_grid.py`,
`data/cie_tables.py`) is copied, not rederived, from the sibling "Inverse
Spectral Rendering" repo. Treat these as fixed dependencies; changes to them
should be made upstream and re-synced, not patched locally.

**The 9 ReSTIR-specific modules** are built in dependency/unblocking order,
not alphabetical or file order — this order is load-bearing and documented in
`.claude/memory/infra_checklist.md` and the README. Each module maps to one
or more items in the A (analytical) / T (toy numerical) / G (graphable
sweep) / V (real-tracer verification) dependency table in
`.claude/ref/forward_paper1_test_suite.md`; `.claude/ref/restir_running_notes.md`
has the full statement of each A-item the code implements.

1. `ris_reservoir.py` — RIS / weighted-reservoir core (streaming WRS
   `update`, MIS-free pairwise `merge`, `contribution_weight` = W).
2. `recorrelation_sampler.py` — recorrelation-lemma sampler
   `p(lambda|lambda') = e(lambda)/integral(e)` and the joint RIS target
   `a(lambda')*L_e(lambda')*G`.
3. `shift_maps.py` — reconnection-validity shift maps `T_{A->B}`
   (identity for non-dispersive path types per A1, genuine Snell reshift +
   Jacobian for dispersive, `(None, 0.0)` at TIR/non-existence).
4. `mis_combine.py` — generalized balance-heuristic MIS
   (`balance_heuristic_weight`, `combine_reservoirs`).
   **Checkpoint: modules 1-4 unlock T1-T11.**
5. `freepath_sampler.py` — volumetric free-path sampler (1D delta tracking).
6. `heterogeneous_lookup.py` — heterogeneous local-lookup / A6 consistency
   trichotomy: `naive_score` / `is_reweight_score` / `fix_local_score`
   reuse strategies, plus `has_support_violation` /
   `lookup_trichotomy_case` for the structural-bias classification. Naive
   is only genuinely biased when candidate generation's species weight
   doesn't track the true target's marginal (e.g. ignores a transmittance
   term) — see the module docstring's "naive's cancellation trap" note
   before changing the toy model in its tests.
   **Checkpoint: modules 5-6 unlock T12-T16.**
7. `temporal_history.py` — temporal history buffer + combine rule
   (reprojection, disocclusion, M-clamping; A9/A10). `HistoryBuffer.reproject`
   is always identity (locked v1 static-geometry scope — no rigid motion).
   `temporal_combine` delegates to `mis_combine.combine_reservoirs` with
   identity shifts between current/history (same pixel/vertex domain); the
   only module-7-specific logic is the Coverage Lemma's `wsum_gen>0` gate
   (drop the history reservoir entirely — `M=0` — rather than let stale
   confidence leak into the MIS denominator) and passing `m_cap` through for
   input-side M-clamping. When testing this module, the unbiased per-frame
   estimator is `combined.wsum` itself, **not**
   `combined.contribution_weight()` (which divides by the combined `M` for
   downstream chaining across further frames, not per-frame correctness) —
   see the module's test file if this trips you up again.
8. `pcg32_rng.py` — production Philox/PCG32 keying scheme (must not be
   plain `numpy.random` / `torch.Generator` — needed for T25's cross-frame
   correlation check specifically).
   **Checkpoint: modules 7-8 unlock T17-T25.**
9. `furnace_canary.py` — Neumann-series analytic reference, for V-tier and
   T-tier z-score/ESS/Rao-Blackwell checks (T3, T14, T22).

Status: modules 1-5 implemented and tested. 6-9 are stubs (signature +
docstring + `raise NotImplementedError` only).

### Design rules that must not be violated

- **MIS combine logic stays out of `Reservoir.merge`.** The generalized
  balance-heuristic MIS combine (module 4) must remain its own standalone
  unit, never inlined into `ris_reservoir.py`. A real bug (T5, dispersive
  hard-cutoff support mismatch) previously lived in this logic when it
  wasn't separated out; keeping it isolated is what makes it independently
  testable and re-verifiable.
- **A missing reconnection is an existence failure, not a zero-weight
  candidate.** Shift maps return `(None, 0.0)` when a reconnection doesn't
  exist (TIR, support mismatch). Downstream code (`balance_heuristic_weight`,
  `combine_reservoirs`) must drop that term from the sum entirely rather than
  substituting a phantom zero — silently zero-filling instead of dropping
  reproduces the same bug class T5 caught.
- **Volumetric shifts are always identity in the locked v1 scope (A2)** —
  don't add non-identity volumetric shift logic without revisiting that
  scope lock.
- Each module's docstring cites which A-item/theorem it implements and which
  T-items depend on it — read the docstring before changing the module's
  math, since the shape of the code is dictated by a specific proof in
  `restir_running_notes.md`, not by API convenience.

### Test style

Tests are point-probes, not fuzzing: exact closed-form checks where possible
(`abs(got - expected) < 1e-12`), Monte Carlo empirical-vs-expected checks with
a stated tolerance where a closed form isn't available (e.g.
`torch.allclose(..., atol=0.01)` over `N=200_000` trials), and one dedicated
probe per edge case named in the module's docstring (e.g. the "nonexistent
shift drops source entirely" case). When adding a module, follow this same
pattern rather than introducing a new test framework or style.
