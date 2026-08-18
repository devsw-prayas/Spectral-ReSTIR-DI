# Spectral ReSTIR DI

Forward Paper 1, companion to "Inverse Spectral Rendering" (Inverse Paper 1).
This repo currently holds the project infrastructure: shared spectral
utilities and the ReSTIR specific modules that the paper's toy numerical
probes (T tier tests) are built on top of.

## Layout

```
src/                      Python package, torch based, float64 throughout
  check_env.py             environment sanity check (torch/scipy/matplotlib, CUDA, dtype)
  snell_jacobian.py         Tier 0, imported from Inverse Spectral Rendering
  cauchy_ior.py             Tier 0, imported from Inverse Spectral Rendering
  spectral_grid.py          Tier 0, imported from Inverse Spectral Rendering
  data/cie_tables.py        Tier 0, imported from Inverse Spectral Rendering
  ris_reservoir.py          module 1, RIS / weighted reservoir core
  recorrelation_sampler.py  module 2, recorrelation lemma sampler
  shift_maps.py             module 3, reconnection validity shift maps
  mis_combine.py            module 4, MIS balance heuristic combine
  freepath_sampler.py       module 5, volumetric free path sampler (delta tracking)
  heterogeneous_lookup.py   module 6, heterogeneous local-lookup / A6 trichotomy
  temporal_history.py       module 7, temporal history buffer + combine (A9/A10)
  pcg32_rng.py              module 8, Philox/PCG32 keying scheme
  furnace_canary.py         module 9, Neumann-series analytic reference
tests/                     one test file per implemented module, plus
                           dedicated test_t<N>_*.py files (T1-T34) and
                           test_a_*.py SymPy proof files (A-series)
results/figures/           output directory for graphable sweeps (G-series)
```

## Build order

The 9 ReSTIR specific modules are built in the order that unblocks the most
toy numerical probes (T tier tests) fastest, not file alphabetical order.
Modules 1 through 4 form the first checkpoint (they unblock T1 through T11),
modules 5 and 6 form the next (T12 through T16), modules 7 and 8 the next
(T17 through T25), then module 9. See the test plan for the full A/T/G/V
dependency table this build order is derived from.

Status as of this commit: all 9 modules are implemented and covered by
passing tests. T1 through T34 (toy numerical probes) and the A-series
(closed-form SymPy proofs) are complete. G-series graphable sweeps: G1/G2
are done; G3 and G8 are intentionally left open (they surfaced a genuine
discrepancy — naive confidence-share is provably unbiased under smooth
separation — rather than a bug to fix).

## Environment

Developed against the `Spectral` conda environment (torch 2.5.1, CUDA 12.4,
float64 default dtype). Run the environment check with:

```
python src/check_env.py
```

## Running tests

Each implemented module has a standalone test file under `tests/` that can
be run directly, for example:

```
python tests/test_ris_reservoir.py
```
