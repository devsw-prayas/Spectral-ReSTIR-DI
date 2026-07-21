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
  heterogeneous_lookup.py   module 6, stub
  temporal_history.py       module 7, stub
  pcg32_rng.py              module 8, stub
  furnace_canary.py         module 9, stub
tests/                     one test file per implemented module
results/figures/           output directory for later graphable sweeps
```

## Build order

The 9 ReSTIR specific modules are built in the order that unblocks the most
toy numerical probes (T tier tests) fastest, not file alphabetical order.
Modules 1 through 4 form the first checkpoint (they unblock T1 through T11),
modules 5 and 6 form the next (T12 through T16), and so on. See the test
plan for the full A/T/G/V dependency table this build order is derived from.

Status as of this commit: modules 1 through 5 are implemented and covered by
passing tests. Modules 6 through 9 are stubbed (signatures and docstrings
only, no logic yet).

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
