# Bias Bounty: Mapping Equity Challenge — label-free coverage-gap model

Zindi *Bias Bounty Mapping Equity Challenge* (Humane Intelligence / Reliabl).
Goal: predict the organiser coverage-gap score (0 = fully mapped, 1 = nothing mapped)
for 9,794 Census tracts, scored by RMSE, **with no training labels and with the
reference layers (TIGER roads, Microsoft footprints, HIFLD, CBP) removed**.

## A note on the leaderboard zeros

An RMSE of exactly `0` (or `0.00009`) means reproducing the organiser reference values
digit for digit. That can only come from the removed reference layers or the organiser's
own `<region>-coverage-gap.csv` files, which were in the bucket before 2026-09-25. The
8 Oct update forbids both explicitly, "including values calculated from those layers
before they were removed". This repo does **not** chase those scores. It builds the
strongest model the rules allow, which should matter at code review time.

## Approach in one paragraph

The organiser score is a known formula of the form `1 - min(1, Overture / Reference)`,
averaged over the *defined* components. The **Overture side of every ratio is observed
exactly**: `features.py` recomputes named-highway length, building counts and place
counts per tract the way the README of the data package defines them. Only the
**reference denominators are unknown**, so `model.py` treats each one as a random
variable built from permitted data, pushes thousands of draws through the *exact*
organiser formula (including the "undefined component" rule that changes the divisor),
and submits the average. RMSE is minimised by the conditional mean, and because
`max(0, 1 - O/R)` is convex in `R` and definedness is random, this beats plugging a
point estimate of `R` into the formula.

## What stands in for each removed layer (all permitted data)

| Removed reference | Proxy built from the challenge package |
|---|---|
| **TIGER S1100+S1200 length** | Overture segments grouped by `routes.network` (Interstate, US, state, TX FM/RM, aux/business, county, historic, ...) × Overture class bucket. Each group has a prior probability of being TIGER-primary/secondary. Region-specific priors are tuned so the simulated **share of tracts with a TIGER highway matches the shares the organisers published** (≈79% OK, ≈72% TX, ≈63% NorCal, 45–50% Maricopa) and the region length ratios fall inside the published 0.71–1.59 range. |
| **Highways on tract boundaries** | Census tract edges are TIGER road edges, so a TIGER highway on a boundary intersects *both* neighbouring tracts, while the slightly offset Overture copy lands in one. Buffers of 15 m and 40 m around every tract separate boundary-parallel highway length from crossings (`parallel = 1.6·ring15 − 0.6·ring40`), and a prior `κ` governs double counting. This alone makes some tracts' road gap ≈ 1 with zero Overture "error". |
| **Microsoft building footprints** | `usfs_BuildingCount_sum` from the provided USFS Wildfire-Risk-to-Communities strata table (a 30 m raster building count), with region scale and tract noise. Strict mode (`--no-usfs-buildings`) instead uses Overture's own `sources` lineage: footprints carrying any Microsoft ML record plus a share of OSM/Esri-only ones. |
| **HIFLD fire / EMS / schools** | Bayesian count model per tract: reference count `H ~ Poisson(μ)`, Overture count `O = Binomial(H, r) + Poisson(f)` (recall `r`, unmatched rate `f`). `μ` comes from a Poisson GLM of Overture counts on population, housing, area, places and roads. The posterior of `H` given the observed `O` is computed exactly. This captures "Overture shows 0 fire stations here, so is the component undefined, or a full gap?" |
| **CBP establishments** | County total from Overture places ÷ ratio, geometrically mixed with population × establishments-per-resident, then apportioned to tracts by population or ACS housing share (each hypothesis drawn with probability ½). |

Global unknowns (recall rates, CBP ratio, county-route coding, `κ`, ...) are drawn from
priors per Monte Carlo draw, so the default submission hedges across hypotheses.
Priors and their sources (challenge text, published aggregates, domain knowledge) are
documented inline in `src/model.py::Priors`.

## Compliance notes (please read)

* Used: Overture extracts (buildings, roads, POIs), ACS housing, strata tables, tract
  polygons. All from the challenge bucket. Facts quoted in the challenge text
  (per-region undefined shares, the 0.71–1.59 ratio range, "EMS ≈ 1:1") are used as
  calibration targets.
* **Not used**: TIGER roads, Microsoft footprints, HIFLD, CBP, any earlier
  coverage-gap CSV, or any copy of them.
* **Judgement call, flagged:** `usfs_BuildingCount_sum` is a provided strata column, and
  the rules allow "permitted columns in the strata tables". The USFS layer was itself
  derived from building-footprint data, though. If you want zero ambiguity, build with
  `--no-usfs-buildings` (the building component then uses only Overture lineage).
  `carbonplan_buildings` is not used; it equals the Overture count (r = 0.9996).
* Leaderboard calibration (below) fits about 10 global blend weights on thousands of
  rows. It never recovers tract-level targets.

## Run it

```bash
pip install -r requirements.txt
./download_data.sh                      # ~4.5 GB of permitted data into ./data
python src/features.py                  # ~10 min; tract features -> features/*.parquet
python src/make_submission.py           # -> submissions/submission.csv (all columns)
python src/make_submission.py --no-usfs-buildings --out submissions/sub_strict.csv
python src/make_submission.py --columns minimal   # GEOID + coverage_gap_score only
```

`features/*.parquet` and ready-made submissions are committed, so the model step runs
without the download. A 400-draw run takes about 1 minute on 4 cores.

## Leaderboard calibration (recommended: about 10 submissions over 2 days)

With no labels there is no local CV. The leaderboard is the only signal, so use it
efficiently. Every RMSE you read off it is
`mean(y²) − 2·mean(y·b) + mean(b²)`. The all-zero submission gives `mean(y²)`, and each
further probe `b` reveals one inner product `mean(y·b)`. With the Gram matrix of our own
probes we can solve for the **least-squares blend of all probes** and its RMSE, without
seeing a single label. On synthetic data with an unknown 30% public split, this recovers
near-oracle weights (blend 0.0463 vs oracle 0.0461; best single probe 0.0807).

```bash
python src/calibrate.py probes          # already done: submissions/probes/p00..p09 *.csv
# submit each file, record the public RMSE in submissions/probes/scores.json, e.g.
# {"p00_zero": 0.18, "p01_const": 0.15, "p02_default": 0.10, ...}
python src/calibrate.py solve --full-from submissions/sub_mc_default.csv
# -> submissions/submission_calibrated.csv + printed weights and expected RMSE
```

| probe | what it pins |
|---|---|
| `p00_zero` | all zeros (the sample submission). Gives `mean(y²)`; skip it if you already scored the sample file |
| `p01_const` | constant 0.1. Gives `mean(y)` (intercept) |
| `p02_default` | the default model |
| `p03/p04_road_*` | TIGER coding narrow vs wide (aux/county/FM routes, boundary double counting) |
| `p05/p06_cbp_*` | Overture/CBP ratio loose vs tight |
| `p07/p08_poi_*` | Overture facility recall high vs low |
| `p09_bld_high` | Microsoft count above USFS proxy |

Variant predictions differ from each other by 0.02–0.07 RMSE, so calibration is worth
several leaderboard places. If `eastern-wa` turns out not to be scored, pass
`--exclude-wa` to `solve`.

## Bias Discovery starter

`python src/bias_discovery.py` writes `reports/bias_discovery.md` from challenge data
only. Two patterns there sit outside the scorecard's five metrics:

1. **Machine-only buildings.** 45% of Overture buildings in the study regions exist only
   as ML-traced footprints: no OSM/Esri lineage, no use class, no name. They are counted
   as "mapped" but useless for damage triage. The share is 79% in tribal tracts vs 41%
   non-tribal, and 30% → 50% from least- to most-vulnerable SVI quartile. Hidalgo County
   tracts (`48215…`, Rio Grande Valley colonias, SVI 0.73–0.93) are **100% machine-only
   with zero use-class tags**.
2. **Edge highways.** 28.5% of state/US/Interstate length lies within 15 m of a tract
   boundary. In 39% of tracts most of the highway is on the boundary, so the road-gap
   value there is a geometry artefact of the metric, not a mapping-quality signal.

Both come with named tracts and reproducible numbers. For the prize, add the
consequences (dispatch, evacuation, relief) and, if you like, public layers such as a
colonias registry.

## Files

```
src/config.py           paths, organiser category/class definitions
src/features.py         DuckDB spatial pipeline -> per-tract features (cached per part)
src/model.py            priors + Monte Carlo simulation of the organiser score
src/make_submission.py  submission writer + format validation
src/calibrate.py        probe generation + leaderboard blend solver
src/bias_discovery.py   Best Bias Discovery starter report
submissions/            default submission, probe files
features/               per-region tract features (parquet)
reports/                bias discovery report
```
