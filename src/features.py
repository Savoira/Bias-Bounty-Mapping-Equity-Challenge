"""Tract-level feature extraction from the *permitted* challenge data only.

Inputs (all from the challenge bucket, see download_data.sh):
  reference/<region>/<region>-overture-{buildings,roads,pois}.parquet
  reference/<region>/<region>-census-acs-housing.parquet
  strata/<region>/<region>-census-tracts.parquet   (tract polygons)
  strata/<region>/<region>-strata-tract-table.parquet

Nothing here touches TIGER roads, Microsoft footprints, HIFLD or CBP.

The Overture side of every organiser ratio (road length by class, building count,
place counts by category) is computed exactly as the challenge README describes, so
it is *observed*, not estimated. Everything that stands in for a removed reference
layer is a feature for the model in model.py.

Usage:  python src/features.py [region ...] [--refresh road-pieces,pois,buildings]
"""
from __future__ import annotations

import sys
import time

import duckdb
import numpy as np
import pandas as pd

from config import (FEAT_DIR, NAMED_CLASSES, REGIONS, RING_WIDTHS, SCHOOL_CATS,
                    ref_path, sample_path, strata_path)

STATES = ("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO "
          "MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC").split()
STATE_RE = "(" + "|".join(STATES) + ")"

# Route-network groups, in priority order (a segment carrying several routes takes the
# lowest code). These are the candidates for TIGER S1100/S1200; how much of each group
# TIGER actually codes as primary/secondary is learned in model.py.
ROUTE_GROUPS = {1: "I", 2: "TOLL", 3: "US", 4: "ST", 5: "FM", 6: "IAUX", 7: "USAUX",
                8: "STAUX", 9: "CO", 10: "HIST", 11: "OTH", 12: "NONE"}

ROUTE_CASE = f"""
CASE
  WHEN n = 'US:I' THEN 1
  WHEN regexp_matches(n, '^US:{STATE_RE}:(Toll|Turnpike)') THEN 2
  WHEN n = 'US:US' THEN 3
  WHEN regexp_matches(n, '^US:{STATE_RE}$') THEN 4
  WHEN regexp_matches(n, '^US:TX:(FM|RM|PR|RE|NASA|UR)$') THEN 5
  WHEN regexp_matches(n, '^US:I:') THEN 6
  WHEN regexp_matches(n, '^US:US:') AND n <> 'US:US:Historic' THEN 7
  WHEN regexp_matches(n, '^US:{STATE_RE}:(Loop|Spur|Business|Truck|Alternate|Bypass|Connector|Beltway|Express|Park|Scenic)') THEN 8
  WHEN regexp_matches(n, '^US:{STATE_RE}:') THEN 9
  WHEN n = 'US:US:Historic' THEN 10
  ELSE 11
END"""


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.sql("INSTALL spatial; LOAD spatial;")
    # Layers are OGC:CRS84 (lon/lat). Without this DuckDB 2.x refuses / mis-orders axes.
    con.sql("SET geometry_always_xy = true;")
    return con


def load_tracts(con, region: str) -> None:
    con.sql(f"""
        CREATE OR REPLACE TABLE tr84 AS
        SELECT GEOID, COUNTYFP, STATEFP, geometry AS g
        FROM '{strata_path(region, "census-tracts")}'
    """)
    rings = ",\n".join(
        f"ST_Buffer(g, {w}) AS gp{w}, ST_Buffer(g, -{w}) AS gm{w}" for w in RING_WIDTHS)
    con.sql(f"""
        CREATE OR REPLACE TABLE tr AS
        SELECT GEOID, g, {rings}, ST_Area(g) AS area_m2, ST_Length(ST_Boundary(g)) AS perim_m
        FROM (SELECT GEOID, ST_Transform(g, 'OGC:CRS84', 'EPSG:5070') AS g FROM tr84)
    """)


def road_features(con, region: str) -> pd.DataFrame:
    named = ", ".join(f"'{c}'" for c in NAMED_CLASSES)
    con.sql(f"""
        CREATE OR REPLACE TABLE rd AS
        WITH seg AS (
            SELECT id, class, subclass, routes, geometry FROM '{ref_path(region, "overture-roads")}'
        ), grp AS (
            SELECT id, min({ROUTE_CASE}) AS grp
            FROM (SELECT id, coalesce(r.network, 'REF:' || r.ref) AS n FROM seg, unnest(routes) AS t(r))
            GROUP BY id
        )
        SELECT s.id, s.class, coalesce(s.subclass, '') AS subclass,
               coalesce(g.grp, 12) AS grp,
               s.class IN ({named}) AS is_named,
               ST_Transform(s.geometry, 'OGC:CRS84', 'EPSG:5070') AS g
        FROM seg s LEFT JOIN grp g USING (id)
        WHERE s.class IN ({named}, 'tertiary') OR g.grp IS NOT NULL
    """)
    w_big = max(RING_WIDTHS)
    ring_cols = ",\n".join(
        f"ST_Length(ST_Intersection(r.g, t.gp{w})) AS lp{w}, "
        f"ST_Length(ST_Intersection(r.g, t.gm{w})) AS lm{w}" for w in RING_WIDTHS)
    con.sql(f"""
        CREATE OR REPLACE TABLE rd_tr AS
        SELECT t.GEOID, r.class, r.subclass, r.grp, r.is_named,
               ST_Length(ST_Intersection(r.g, t.g)) AS lin,
               {ring_cols}
        FROM rd r JOIN tr t ON ST_Intersects(r.g, t.gp{w_big})
    """)
    return con.sql("SELECT * FROM rd_tr").df()


def aggregate_roads(df: pd.DataFrame) -> pd.DataFrame:
    """Per-tract road features from the (segment x tract) pieces table."""
    out = {}
    link = df.subclass.eq("link")
    nm = df[df.is_named]
    # Exact Overture side of the organiser road ratio (links included: Overture puts
    # motorway_link etc. in class motorway/trunk/..., subclass link).
    out["o_named"] = nm.groupby("GEOID").lin.sum()
    out["o_named_nolink"] = nm[~nm.subclass.eq("link")].groupby("GEOID").lin.sum()
    out["o_link"] = nm[nm.subclass.eq("link")].groupby("GEOID").lin.sum()
    for c in NAMED_CLASSES + ("tertiary",):
        out[f"o_{c}"] = df[df["class"] == c].groupby("GEOID").lin.sum()
    for w in RING_WIDTHS:
        # Overture named-highway length inside the tract but within w of its edge: the
        # part whose side-of-boundary relative to TIGER is a coin flip.
        out[f"o_named_inner{w}"] = (nm.lin - nm[f"lm{w}"]).groupby(nm.GEOID).sum()
    # Candidate TIGER S1100/S1200 pieces: route group x Overture class bucket
    # (mw = motorway main line, n = other named class, o = tertiary and below).
    # Ramps are left out: TIGER codes them S1630, not S1100/S1200.
    bucket = np.where(df["class"].eq("motorway"), "mw", np.where(df.is_named, "n", "o"))
    nl = df[~link].assign(b=bucket[~link.to_numpy()])
    for code, name in ROUTE_GROUPS.items():
        for b in ("mw", "n", "o"):
            sub = nl[(nl.grp == code) & (nl.b == b)]
            if sub.empty:
                continue
            gid = sub.GEOID
            out[f"r_{name}_{b}_in"] = sub.lin.groupby(gid).sum()
            for w in RING_WIDTHS:
                out[f"r_{name}_{b}_r{w}"] = (sub[f"lp{w}"] - sub.lin).groupby(gid).sum()
            # inside the tract but within 15 m of its edge
            out[f"r_{name}_{b}_i15"] = (sub.lin - sub.lm15).groupby(gid).sum()
    res = pd.DataFrame(out)
    res.index.name = "GEOID"
    return res


def building_features(con, region: str) -> pd.DataFrame:
    return con.sql(f"""
        WITH b AS (
            SELECT ST_Centroid(geometry) AS c,
                   ST_Area(geometry) * 12392029030.5 * cos(radians(ST_Y(ST_Centroid(geometry)))) AS a,
                   sources[1].dataset AS ds,
                   list_contains(list_transform(sources, lambda s: s.dataset), 'Microsoft ML Buildings') AS ms_any,
                   list_contains(list_transform(sources, lambda s: s.dataset), 'USGS Lidar') AS lidar_any,
                   sources[1].confidence AS conf,
                   subtype, height, names.primary AS nm,
                   sources[1].update_time AS ut
            FROM '{ref_path(region, "overture-buildings")}'
        )
        SELECT t.GEOID,
               count(*) AS b_n,
               count(*) FILTER (WHERE ds = 'OpenStreetMap') AS b_osm,
               count(*) FILTER (WHERE ds = 'Microsoft ML Buildings') AS b_ms,
               count(*) FILTER (WHERE ds = 'Esri Community Maps') AS b_esri,
               count(*) FILTER (WHERE ds = 'Google Open Buildings') AS b_google,
               count(*) FILTER (WHERE ms_any) AS b_ms_any,
               count(*) FILTER (WHERE lidar_any) AS b_lidar_any,
               count(*) FILTER (WHERE ds = 'Microsoft ML Buildings' AND conf < 0.9) AS b_ms_lowconf,
               count(*) FILTER (WHERE subtype IS NOT NULL) AS b_typed,
               count(*) FILTER (WHERE subtype = 'residential') AS b_resid,
               count(*) FILTER (WHERE subtype IN ('outbuilding','agricultural')) AS b_outb,
               count(*) FILTER (WHERE nm IS NOT NULL) AS b_named,
               count(*) FILTER (WHERE a < 40) AS b_small,
               count(*) FILTER (WHERE a >= 40 AND a < 120) AS b_mid,
               count(*) FILTER (WHERE a >= 500) AS b_large,
               sum(a) AS b_area,
               median(a) AS b_area_med,
               count(*) FILTER (WHERE ds = 'OpenStreetMap' AND ut >= '2024-01-01') AS b_osm_recent
        FROM b JOIN tr84 t ON ST_Within(b.c, t.g)
        GROUP BY t.GEOID
    """).df().set_index("GEOID")


def poi_features(con, region: str) -> pd.DataFrame:
    schools = ", ".join(f"'{c}'" for c in SCHOOL_CATS)
    school_cols = ",\n".join(
        f"count(*) FILTER (WHERE cat = '{c}') AS p_sch_{c}" for c in SCHOOL_CATS)
    return con.sql(f"""
        WITH p AS (
            SELECT geometry AS c, categories.primary AS cat, categories.alternate AS alt,
                   confidence AS conf, operating_status AS st, sources[1].dataset AS ds,
                   basic_category AS bc
            FROM '{ref_path(region, "overture-pois")}'
        )
        SELECT t.GEOID,
               count(*) AS p_all,
               count(*) FILTER (WHERE st IS DISTINCT FROM 'permanently_closed') AS p_notclosed,
               count(*) FILTER (WHERE cat IS NOT NULL) AS p_cat,
               count(*) FILTER (WHERE conf >= 0.8) AS p_conf80,
               count(*) FILTER (WHERE conf < 0.5) AS p_conf_lt50,
               count(*) FILTER (WHERE cat = 'fire_department') AS p_fire,
               count(*) FILTER (WHERE cat = 'fire_department' AND conf >= 0.8) AS p_fire_hc,
               count(*) FILTER (WHERE cat <> 'fire_department' AND list_contains(alt, 'fire_department')) AS p_fire_alt,
               count(*) FILTER (WHERE cat = 'fire_protection_service') AS p_fire_prot,
               count(*) FILTER (WHERE cat = 'ambulance_and_ems_services') AS p_ems,
               count(*) FILTER (WHERE cat <> 'ambulance_and_ems_services' AND list_contains(alt, 'ambulance_and_ems_services')) AS p_ems_alt,
               count(*) FILTER (WHERE cat IN ({schools})) AS p_school,
               {school_cols},
               count(*) FILTER (WHERE cat NOT IN ({schools}) AND list_has_any(alt, [{schools}])) AS p_school_alt,
               count(*) FILTER (WHERE cat IN ('preschool','charter_school','montessori_school','religious_school','day_care_preschool')) AS p_school_adj,
               count(*) FILTER (WHERE cat = 'hospital') AS p_hospital,
               count(*) FILTER (WHERE cat IN ('police_department')) AS p_police,
               count(*) FILTER (WHERE cat IN ('church_cathedral','religious_organization','church')) AS p_church,
               count(*) FILTER (WHERE cat IN ('restaurant','fast_food_restaurant','mexican_restaurant','pizza_restaurant','coffee_shop')) AS p_food,
               count(*) FILTER (WHERE cat IN ('real_estate_agent','real_estate_service','professional_services','contractor','insurance_agency')) AS p_prof,
               count(*) FILTER (WHERE cat IN ('park','lake','landmark_and_historical_building','river','mountain')) AS p_nature,
               count(*) FILTER (WHERE ds = 'meta') AS p_src_meta,
               count(*) FILTER (WHERE ds = 'Microsoft') AS p_src_ms,
               count(*) FILTER (WHERE ds = 'Foursquare') AS p_src_fsq,
               count(*) FILTER (WHERE ds = 'BrightQuery') AS p_src_bq,
               count(*) FILTER (WHERE ds = 'AllThePlaces') AS p_src_atp,
               avg(conf) AS p_conf_mean
        FROM p JOIN tr84 t ON ST_Within(p.c, t.g)
        GROUP BY t.GEOID
    """).df().set_index("GEOID")


STRATA_COLS = [
    "GEOID", "STATEFP", "COUNTYFP", "ALAND", "pop_total", "pop_urban", "pct_urban", "ur_class",
    "svi_overall", "svi_socioeconomic", "svi_household", "svi_minority", "svi_housing_transport",
    "cvi_overall", "cvi_baseline_infrastructure", "ruca_primary", "ruca_secondary", "rucc_2023",
    "rucc_metro", "nchs_2013", "tribal_any", "tribal_pct", "tribal_legal", "tribal_statistical",
    "usfs_BuildingCount_sum", "usfs_BuildingCover_mean", "usfs_BuildingDensity_mean",
    "usfs_HUCount_sum", "usfs_PopCount_sum", "usfs_WHP_mean", "usfs_RPS_mean",
    "carbonplan_buildings", "usdm_summer_dsci", "epht_heat_days_summer", "ghcn_any_nearest_km",
    "mtbs_wildfire_burned_pct_land", "fod_fires", "uhe_covered",
]


def build_region(region: str) -> pd.DataFrame:
    t0 = time.time()
    con = connect()
    load_tracts(con, region)
    sample = pd.read_csv(sample_path(region), dtype={"GEOID": str})
    strata = pd.read_parquet(strata_path(region, "strata-tract-table"))
    strata = strata[[c for c in STRATA_COLS if c in strata.columns]].set_index("GEOID")
    acs = con.sql(f"SELECT GEOID, housing_units FROM '{ref_path(region, 'census-acs-housing')}'"
                  ).df().set_index("GEOID")
    geo = con.sql("SELECT GEOID, area_m2, perim_m FROM tr").df().set_index("GEOID")
    print(f"[{region}] tracts loaded {time.time()-t0:.0f}s", flush=True)
    cache = FEAT_DIR / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    def cached(part, fn):
        f = cache / f"{region}-{part}.parquet"
        if f.exists() and part not in REFRESH:
            return pd.read_parquet(f)
        res = fn(con, region)
        res.to_parquet(f)
        print(f"[{region}] {part} {time.time()-t0:.0f}s", flush=True)
        return res

    roads = aggregate_roads(cached("road-pieces", road_features))
    pois = cached("pois", poi_features)
    blds = cached("buildings", building_features)

    df = geo.join([strata, acs, roads, blds, pois], how="left")
    num = df.select_dtypes("number").columns
    fill0 = [c for c in num if c.startswith(("o_", "r_", "b_", "p_")) and c not in
             ("b_area_med", "p_conf_mean")]
    df[fill0] = df[fill0].fillna(0.0)
    df = df.copy()
    df["region"] = region
    df["county"] = df.index.str[:5]
    df["scored"] = df.index.isin(sample.GEOID)
    return df.reset_index()


REFRESH: set[str] = set()


def main(regions):
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    for r in regions:
        df = build_region(r)
        df.to_parquet(FEAT_DIR / f"{r}.parquet", index=False)
        print(f"[{r}] wrote {len(df)} tracts x {df.shape[1]} cols", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--refresh" in args:  # --refresh road-pieces,pois,buildings
        i = args.index("--refresh")
        REFRESH.update(args[i + 1].split(","))
        del args[i:i + 2]
    main(args or REGIONS)
