+++
id = "LF2"
title = "HEALPix on JAX"
category = "Structure"
step = "2a"
phase = "2a.X"
status = "ready"
decide = []
depends = ["LF1"]
changes_values = "no; minor"
issue = 0
+++

## Goal

HEALPix statistics computed and differentiated on JAX, and declared stable with the JAX backend (LF1).

## Inputs

Part III §17.2, §17.4, §17.5; Part II §9.3 ("HEALPix in step 1"), §12.3 (`gather_weighted`, `segment_sum`); Part I §3.1, rule 4 (large constants are arguments); maths M-CONV-HPX, M-DOWN-HPX, M-MASK-HPX; the cards LF1, LF62.
Modules: `src/estel/backend/jax/_primitives.py` (`gather_weighted`, `segment_sum`, implemented since T1.3), the HEALPix data and tables of step 1 (T5.1), the HEALPix tests.

## Tasks

1. The HEALPix tests on JAX in float32 and float64, reference and second-order tests included (the full tier runs them from step 1): fix what fails in L0; a fix needed in L1 or L2 is a bug of step 1 (§17.2).
2. The HEALPix cases of the benchmark of step 2 (LF62) on JAX: compile time within budget; the peak memory of `gather_weighted`, whose `take` builds every stencil value before the sum unless XLA fuses it, against PyTorch's; a lighter gather if over budget.
3. Declared stable with LF1, in its last task.

## Done when

Every HEALPix test passes on JAX in both dtypes, within the row "PyTorch vs JAX" of `tests/tolerances.toml`; the second-order tests pass on JAX for HEALPix; the HEALPix cases of LF62 compile within budget on JAX. HEALPix spectra are not part of it (LF24).

## Feature

HEALPix on JAX

## Why

HEALPix on every backend

## Builds on (reserved in 1.0.0)

`gather_weighted`, `segment_sum`, the NumPy tables ported in step 1

## To decide

—

## Details

§17.4

## Source

C9

## Step before 20 November 2026

?, with LF1
