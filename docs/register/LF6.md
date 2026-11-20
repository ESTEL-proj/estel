+++
id = "LF6"
title = "Low-pass filters in every geometry and S3/S4"
category = "New feature"
step = "2a"
phase = "2a.1"
status = "decision"
decide = ["S3 and S4: low-pass as last filter only, or also first", "half-pixel shift without the final downsampling (M-LP2)", "how the plan describes the low-pass filter and the centring"]
depends = ["LF22"]
changes_values = "yes (new default); reference bump; major decided with the maintainers"
issue = 0
+++

## Feature

Low-pass filters in every geometry and in S3/S4. At the last level the low-pass filter is applied without STL's final downsampling, which removes a size constraint (OP-34); this is a traced change of reference values. It is realised with the smoothing kernel; a dedicated low-pass kernel φ remains possible

## Why

the largest scales enter every statistic

## Builds on (reserved in 1.0.0)

index sets of the plan

## To decide

which combinations in S3 and S4: the low-pass filter as the last filter only (+0.3 to +3.5 % coefficients) or also as a first filter (+38 to +112 %, §20.1); the half-pixel shift of the even smoothing kernel without the final downsampling (maths_step2 M-LP2); how the plan describes the low-pass filter and the centring (§19.12, point 4)

## Details

§20.1, §20.10, §20.11

## Source

—

## Step before 20 November 2026

2a; completed by LF22 (2b)
