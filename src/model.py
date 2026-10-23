"""Monte Carlo estimate of the organiser coverage-gap score per tract.

No tract labels exist, so this is not a supervised model. It is a generative model of
how the organiser score was built (README of the data package):

    road     = 1 - min(1, Overture_named_len / TIGER_S1100+S1200_len)   (undefined if TIGER = 0)
    building = 1 - min(1, Overture_buildings  / Microsoft_footprints)    (undefined if MS = 0)
    poi      = mean_defined( hifld_half, cbp_half )
        hifld_half = mean_defined( fire, ems, schools ), each 1 - min(1, O/H)
        cbp_half   = 1 - min(1, Overture_places / CBP_establishments)
    score    = mean of the defined components

The Overture numerators are observed exactly (features.py replicates the organiser
definitions). Each removed reference denominator is a random variable whose
distribution is built from permitted data only (Overture route refs, Overture source
lineage, strata columns, ACS). We draw the denominators, push every draw through the
exact organiser formula -- including the "undefined component" rule, which makes the
divisor random -- and average.

Because RMSE is minimised by the conditional mean, E[score | permitted data] is the
right thing to submit, and it is NOT the same as plugging a point estimate of the
denominators into the formula (max(0, 1 - O/R) is convex in R, and definedness changes
which components are averaged). That gap is where a plug-in approach loses.

Global parameters that the data cannot pin down (how TIGER codes each route network,
recall of Overture fire stations, CBP/Overture ratio, ...) are themselves drawn from
priors, so the prediction is a hedge across hypotheses. calibrate.py can narrow them
with leaderboard feedback.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.linear_model import PoissonRegressor

from config import FEAT_DIR, REGIONS

ROAD_GROUPS = ["I", "TOLL", "US", "ST", "FM", "IAUX", "USAUX", "STAUX", "CO", "HIST", "OTH", "NONE"]


# ----------------------------------------------------------------------------- priors
# Each prior is (low, high) of a uniform draw, per region per Monte Carlo draw, unless
# noted. Values were set from (a) the challenge text -- e.g. the share of tracts with
# no named highway per region, the 0.71-1.59 Overture/TIGER length ratio range, EMS
# "close to 1:1", hospitals excluded because Overture is 12x HIFLD -- and (b) domain
# knowledge of how TIGER, OSM and the Overture places feed are built. See README.md.
@dataclass
class Priors:
    # P(TIGER codes a route of this group as S1100/S1200), drawn per tract x group
    road_w: dict = field(default_factory=lambda: {
        "I": (0.95, 1.0), "TOLL": (0.85, 1.0), "US": (0.92, 1.0), "ST": (0.85, 1.0),
        "FM": (0.55, 1.0), "IAUX": (0.4, 0.9), "USAUX": (0.3, 0.8), "STAUX": (0.3, 0.8),
        "CO": (0.15, 0.6), "HIST": (0.1, 0.5), "OTH": (0.0, 0.2), "NONE": (0.0, 0.08),
    })
    # Region-specific overrides. Set so the simulated share of tracts with a TIGER
    # highway matches the shares published in the challenge text (no TIGER needed):
    # OK ~79%, TX ~72%, NorCal ~63%, Maricopa 45-50%; and so the region Overture/TIGER
    # length ratios land inside the published 0.71-1.59 range (OK needs county routes).
    road_w_region: dict = field(default_factory=lambda: {
        "eastern-ok": {"CO": (0.4, 0.8), "HIST": (0.3, 0.8), "NONE": (0.05, 0.15)},
        "northern-ca": {"CO": (0.15, 0.45), "HIST": (0.2, 0.6)},
        "south-central-tx": {"FM": (0.25, 0.6), "CO": (0.0, 0.2), "STAUX": (0.1, 0.4),
                             "USAUX": (0.3, 0.7), "NONE": (0.0, 0.03)},
        "maricopa-az": {"CO": (0.2, 0.6), "HIST": (0.3, 0.8), "NONE": (0.0, 0.05)},
    })
    # multiplier on the weights of every non-core group (FM, aux, county, historic,
    # other, unsigned); used by the calibration probes to tilt all hypotheses at once
    road_aux_mult: float = 1.0
    # motorway main line of any route group (incl. unsigned) -> limited access -> S1100
    road_w_mw: tuple = (0.85, 1.0)
    # unsigned tertiary roads
    road_w_none_o: tuple = (0.0, 0.02)
    # P(a highway running along the tract edge is counted in BOTH neighbouring tracts)
    road_kappa: tuple = (0.4, 1.0)
    # multiplicative geometry noise (TIGER vs OSM line work) on each group length
    road_sigma: tuple = (0.05, 0.2)

    # Microsoft footprints / USFS WRC building count (strata column), region scale
    bld_scale: tuple = (0.95, 1.06)
    bld_sigma: tuple = (0.04, 0.12)
    # fallback when the USFS column is not used: share of non-MS-lineage Overture
    # buildings that also exist in the Microsoft layer
    bld_alpha: tuple = (0.6, 1.0)

    # POI types: Overture recall r of reference facilities (in the right tract) and
    # share phi of Overture POIs of that category that have no reference counterpart
    # Overture fire_department is 7-15 per 100k people here vs ~15 reference stations
    # per 100k nationally (more in rural OK): Overture under-counts -> r below 1.
    fire_r: tuple = (0.40, 0.80)
    fire_phi: tuple = (0.10, 0.35)
    ems_r: tuple = (0.45, 0.85)
    ems_phi: tuple = (0.15, 0.45)
    # Overture school categories are 5-8 per 10k people vs ~3-4 reference schools per
    # 10k: Overture over-counts (duplicates across sources) -> large phi.
    school_r: tuple = (0.75, 0.95)
    school_phi: tuple = (0.35, 0.60)
    # reference-count dispersion around the fitted rate (log-normal sd)
    poi_sigma: tuple = (0.2, 0.5)

    # CBP: county establishments = county Overture places / R, apportioned to tracts
    # Overture places run 45-53 per 1k people here; CBP is ~20-26 establishments per 1k.
    cbp_ratio: tuple = (1.6, 2.6)
    # alternative county estimate: establishments per resident; mix weight on the
    # places-based estimate (log space)
    cbp_pop_rate: tuple = (0.019, 0.027)
    cbp_mix: tuple = (0.3, 0.9)
    cbp_county_sigma: tuple = (0.05, 0.25)
    cbp_tract_sigma: tuple = (0.05, 0.35)
    cbp_share_pop_prob: float = 0.5        # else ACS housing-unit share

    def update(self, d: dict) -> "Priors":
        for k, v in d.items():
            if k == "road_w":
                self.road_w.update({g: tuple(x) for g, x in v.items()})
            elif k == "road_w_region":
                for reg, gw in v.items():
                    self.road_w_region.setdefault(reg, {}).update({g: tuple(x) for g, x in gw.items()})
            else:
                setattr(self, k, tuple(v) if isinstance(v, list) else v)
        return self


# ----------------------------------------------------------------------------- data
def load_features(regions=REGIONS, scored_only=True) -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(FEAT_DIR / f"{r}.parquet") for r in regions],
                   ignore_index=True)
    # route groups absent from a region are absent columns, i.e. zero length
    rcols = [c for c in df.columns if c.startswith(("r_", "o_"))]
    df[rcols] = df[rcols].fillna(0.0)
    if scored_only:
        df = df[df.scored].reset_index(drop=True)
    return df


ROAD_BUCKETS = ["mw", "n", "o"]   # motorway main line / other named class / tertiary & below


def road_lengths(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per tract x route group x class bucket: length inside the tract, and the
    boundary-parallel length just outside it (a highway TIGER draws ON the shared
    boundary, which the organiser intersection counts in both tracts)."""
    n = len(df)
    L_in = np.zeros((n, len(ROAD_GROUPS), len(ROAD_BUCKETS)))
    L_par = np.zeros_like(L_in)
    for j, g in enumerate(ROAD_GROUPS):
        for k, b in enumerate(ROAD_BUCKETS):
            c = f"r_{g}_{b}"
            if f"{c}_in" not in df:
                continue
            L_in[:, j, k] = df[f"{c}_in"].to_numpy()
            r15, r40 = df[f"{c}_r15"].to_numpy(), df[f"{c}_r40"].to_numpy()
            # ring15 = parallel + 15*crossings, ring40 = parallel + 40*crossings
            L_par[:, j, k] = np.clip(1.6 * r15 - 0.6 * r40, 0, None)
    return L_in, L_par


def fit_poi_rates(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """E[Overture count | tract covariates] for each facility type (Poisson GLM).

    The fitted rate is the shrinkage target that tells us, for a tract where Overture
    shows zero fire stations, how likely it is that a reference station exists anyway.
    """
    X = pd.DataFrame({
        "lpop": np.log1p(df.pop_total), "lhu": np.log1p(df.housing_units),
        "larea": np.log(df.area_m2 / 1e6 + 0.05), "lpall": np.log1p(df.p_all),
        "lbld": np.log1p(df.b_n), "urb": df.pct_urban.fillna(0),
        "ruca": df.ruca_primary.fillna(1), "lroad": np.log1p(df.o_named / 1000 + df.o_tertiary / 1000),
        "tribal": df.tribal_any.astype(float),
    })
    X = pd.concat([X, pd.get_dummies(df.region, prefix="reg", dtype=float)], axis=1)
    Xs = (X - X.mean()) / X.std().replace(0, 1)
    out = {}
    targets = {"fire": df.p_fire, "ems": df.p_ems, "school": df.p_school}
    for k, y in targets.items():
        m = PoissonRegressor(alpha=1e-3, max_iter=1000).fit(Xs, y)
        out[k] = m.predict(Xs)
    return out


# ----------------------------------------------------------------------------- core
def _u(rng, lo_hi, size=None):
    lo, hi = lo_hi
    return rng.uniform(lo, hi, size)


HMAX = 25


def _log_pois(k, lam):
    return k * np.log(lam) - lam - gammaln(k + 1)


def sample_reference_counts(rng, o, m, r, phi, sigma):
    """Draw reference facility counts H given observed Overture count o.

    H ~ Poisson(mu) with log-normal rate noise; Overture = Binomial(H, r) matched
    + Poisson(f) unmatched. Posterior over H evaluated on 0..HMAX, then sampled.
    """
    n = len(o)
    mu = (1 - phi) * m / r * np.exp(rng.normal(0, sigma, n) - sigma ** 2 / 2)
    f = np.maximum(phi * m, 1e-6)
    mu = np.maximum(mu, 1e-6)
    h = np.arange(HMAX + 1)
    oc = np.minimum(o, HMAX + 10).astype(int)
    logp = _log_pois(h[None, :], mu[:, None])                      # prior on H
    # likelihood P(o | h) = sum_k Binom(k; h, r) Pois(o-k; f)
    lik = np.zeros((n, HMAX + 1))
    r = np.broadcast_to(r, (n,))
    for k in range(0, min(HMAX, oc.max()) + 1):
        hk = np.maximum(h - k, 0)
        comb = np.where(h >= k, gammaln(h + 1) - gammaln(k + 1) - gammaln(hk + 1), -np.inf)
        lb = comb[None, :] + k * np.log(r)[:, None] + hk[None, :] * np.log1p(-r)[:, None]
        rem = oc - k
        lp = np.where(rem >= 0, _log_pois(np.maximum(rem, 0), f), -np.inf)
        lik += np.exp(lb + lp[:, None])
    post = np.exp(logp - logp.max(1, keepdims=True)) * lik
    post /= post.sum(1, keepdims=True)
    cdf = post.cumsum(1)
    u = rng.uniform(size=(n, 1))
    return (u > cdf).sum(1)


def simulate(df: pd.DataFrame, priors: Priors | None = None, n_draws: int = 300,
             seed: int = 0, use_usfs_buildings: bool = True, return_draws: bool = False):
    priors = priors or Priors()
    rng = np.random.default_rng(seed)
    n = len(df)
    reg_codes, reg_names = pd.factorize(df.region)
    n_reg = len(reg_names)

    L_in, L_par = road_lengths(df)
    o_road = df.o_named.to_numpy()
    o_bld = df.b_n.to_numpy().astype(float)
    usfs = df.usfs_BuildingCount_sum.to_numpy()
    usfs_ok = np.isfinite(usfs) & (usfs > 0)
    ms_lin = df.b_ms_any.to_numpy().astype(float)
    o_all = df.p_all.to_numpy().astype(float)
    rates = fit_poi_rates(df)
    o_poi = {"fire": df.p_fire.to_numpy(), "ems": df.p_ems.to_numpy(),
             "school": df.p_school.to_numpy()}

    county = df.county.to_numpy()
    c_codes, _ = pd.factorize(county)
    n_c = c_codes.max() + 1
    pop = df.pop_total.to_numpy().astype(float)
    hu = df.housing_units.fillna(0).to_numpy().astype(float)
    c_places = np.bincount(c_codes, o_all, n_c)
    c_pop = np.bincount(c_codes, pop, n_c)
    c_hu = np.bincount(c_codes, hu, n_c)
    c_reg = np.zeros(n_c, int)
    c_reg[c_codes] = reg_codes

    acc = {k: np.zeros(n) for k in ["score", "w", "road", "road_d", "bld", "bld_d", "poi", "poi_d",
                                    "fire", "fire_d", "ems", "ems_d", "school", "school_d",
                                    "cbp", "cbp_d", "hifld", "hifld_d"]}
    draws = []
    for d in range(n_draws):
        R = lambda key: _u(rng, getattr(priors, key), n_reg)[reg_codes]  # noqa: E731

        # ---------------- road
        T = np.zeros(n)
        kappa = _u(rng, priors.road_kappa)
        sig = _u(rng, priors.road_sigma)
        w_mw = _u(rng, priors.road_w_mw, n_reg)[reg_codes]
        for j, g in enumerate(ROAD_GROUPS):
            w = np.array([_u(rng, priors.road_w_region.get(rn, {}).get(g, priors.road_w[g]))
                          for rn in reg_names])[reg_codes]
            if g not in ("I", "TOLL", "US", "ST"):
                w = np.clip(w * priors.road_aux_mult, 0, 1)
            inc = rng.uniform(size=n) < w
            inc_mw = inc | (rng.uniform(size=n) < w_mw)
            if g == "NONE":
                inc_o = rng.uniform(size=n) < _u(rng, priors.road_w_none_o)
            else:
                inc_o = inc
            lg = L_in[:, j] + kappa * L_par[:, j]
            T += (inc_mw * lg[:, 0] + inc * lg[:, 1] + inc_o * lg[:, 2]) * np.exp(rng.normal(0, sig, n))
        road_d = T > 1.0
        road = np.where(road_d, 1 - np.minimum(1, o_road / np.maximum(T, 1e-9)), 0.0)

        # ---------------- buildings
        if use_usfs_buildings:
            base = np.where(usfs_ok, usfs, ms_lin + 0.8 * (o_bld - ms_lin))
        else:
            base = ms_lin + R("bld_alpha") * (o_bld - ms_lin)
        sb = R("bld_scale")
        sgb = _u(rng, priors.bld_sigma)
        MS = base * sb * np.exp(rng.normal(0, sgb, n) - sgb ** 2 / 2)
        bld_d = MS >= 0.5
        bld = np.where(bld_d, 1 - np.minimum(1, o_bld / np.maximum(MS, 1e-9)), 0.0)

        # ---------------- HIFLD half
        sig_p = _u(rng, priors.poi_sigma)
        parts, defs = {}, {}
        for k in ("fire", "ems", "school"):
            r = _u(rng, getattr(priors, f"{k}_r"), n_reg)[reg_codes]
            phi = _u(rng, getattr(priors, f"{k}_phi"), n_reg)[reg_codes]
            H = sample_reference_counts(rng, o_poi[k], rates[k], r, phi, sig_p)
            defs[k] = H >= 1
            parts[k] = np.where(defs[k], 1 - np.minimum(1, o_poi[k] / np.maximum(H, 1)), 0.0)
        nh = defs["fire"].astype(int) + defs["ems"] + defs["school"]
        hifld_d = nh > 0
        hifld = np.where(hifld_d, (parts["fire"] + parts["ems"] + parts["school"]) / np.maximum(nh, 1), 0.0)

        # ---------------- CBP half
        ratio = _u(rng, priors.cbp_ratio, n_reg)[c_reg]
        sc = _u(rng, priors.cbp_county_sigma)
        mix = _u(rng, priors.cbp_mix)
        rate = _u(rng, priors.cbp_pop_rate, n_reg)[c_reg]
        c_cbp = (np.maximum(c_places / ratio, 1e-9) ** mix * np.maximum(c_pop * rate, 1e-9) ** (1 - mix)
                 * np.exp(rng.normal(0, sc, n_c) - sc ** 2 / 2))
        if rng.uniform() < priors.cbp_share_pop_prob:
            share = np.divide(pop, c_pop[c_codes], out=np.zeros(n), where=c_pop[c_codes] > 0)
        else:
            share = np.divide(hu, c_hu[c_codes], out=np.zeros(n), where=c_hu[c_codes] > 0)
        st = _u(rng, priors.cbp_tract_sigma)
        CBP = c_cbp[c_codes] * share * np.exp(rng.normal(0, st, n) - st ** 2 / 2)
        cbp_d = CBP >= 0.5
        cbp = np.where(cbp_d, 1 - np.minimum(1, o_all / np.maximum(CBP, 1e-9)), 0.0)

        np_ = hifld_d.astype(int) + cbp_d
        poi_d = np_ > 0
        poi = np.where(poi_d, (hifld + cbp) / np.maximum(np_, 1), 0.0)

        # ---------------- composite
        nd = road_d.astype(int) + bld_d + poi_d
        any_d = nd > 0
        score = np.where(any_d, (road + bld + poi) / np.maximum(nd, 1), 0.0)

        acc["score"] += score * any_d
        acc["w"] += any_d
        for k, v, dd in [("road", road, road_d), ("bld", bld, bld_d), ("poi", poi, poi_d),
                         ("fire", parts["fire"], defs["fire"]), ("ems", parts["ems"], defs["ems"]),
                         ("school", parts["school"], defs["school"]), ("cbp", cbp, cbp_d),
                         ("hifld", hifld, hifld_d)]:
            acc[k] += v * dd
            acc[k + "_d"] += dd
        if return_draws:
            draws.append(score.astype(np.float32))

    out = pd.DataFrame({"GEOID": df.GEOID, "region": df.region})
    out["coverage_gap_score"] = acc["score"] / np.maximum(acc["w"], 1)
    for k in ["road", "bld", "poi", "fire", "ems", "school", "cbp", "hifld"]:
        out[f"E_{k}"] = acc[k] / np.maximum(acc[k + "_d"], 1)
        out[f"P_{k}"] = acc[k + "_d"] / n_draws
    if return_draws:
        return out, np.stack(draws, 1)
    return out


def load_priors(path: str | None) -> Priors:
    p = Priors()
    if path:
        with open(path) as fh:
            p.update(json.load(fh))
    return p
