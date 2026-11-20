+++
id = "LF3"
title = "Rust engine (estel-rust), CPU and GPU"
category = "Structure"
step = "2b"
phase = ""
status = "decision"
decide = ["with engine=\"auto\", routing of second-order requests", "scope of the Rust engine after the benchmark (§18.6)"]
depends = []
changes_values = "no; minor; default for a configuration = benchmark-gated minor"
issue = 0
+++

## Feature

Rust engine (`estel-rust`, built from `rust/` in the ESTEL repository; CPU and GPU)

## Why

about 10× faster and 9× lighter in memory at nside 128

## Builds on (reserved in 1.0.0)

`engine=` argument, engine protocol

## To decide

with `engine="auto"`, whether second-order requests are routed to the Python engine while the Rust engine is at order 1; the scope of the Rust engine, after a benchmark of the reorganised Python engine against STL-Dev-MPI (§18.6)

## Details

§18

## Source

C11, C12, C16

## Step before 20 November 2026

2b
