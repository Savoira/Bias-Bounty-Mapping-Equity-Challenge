"""Best Bias Discovery starter: two patterns outside the automated scorecard grid.

Uses only the challenge data (features built by features.py). Writes
reports/bias_discovery.md with the tables and the named tracts behind each finding.

1. Edge highways. Census tract boundaries are drawn on road edges, so a large share of
   highway length sits ON a tract boundary. For those tracts, an Overture-vs-TIGER road
   comparison is decided by which side of the line a few metres of geometry falls, not
   by mapping quality. We measure, from Overture geometry alone, the share of each
   tract's highway length that lies within 15 m of its boundary, and who lives there.

2. Machine-only buildings. Overture's `sources` lineage shows which footprints exist
   only because a Microsoft/Google ML model drew them. Those carry no use class, name or
   height, so they cannot tell a home from a barn, a school or a clinic, which is what
   damage assessment and evacuation planning need. Coverage counts call these tracts
   "fully mapped"; attribute coverage says otherwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import ROOT
from model import ROAD_BUCKETS, ROAD_GROUPS, load_features, road_lengths  # noqa: F401

OUT = ROOT / "reports" / "bias_discovery.md"


def strata(df: pd.DataFrame) -> pd.DataFrame:
    s = pd.DataFrame(index=df.index)
    s["svi_q"] = pd.cut(df.svi_overall, [-0.01, 0.25, 0.5, 0.75, 1.0],
                        labels=["SVI Q1 (least)", "SVI Q2", "SVI Q3", "SVI Q4 (most)"])
    s["tribal"] = np.where(df.tribal_any, "tribal", "non-tribal")
    s["ruca"] = pd.cut(df.ruca_primary, [0, 3, 6, 9, 10],
                       labels=["metro", "micropolitan", "small town", "rural"])
    return s


def edge_highways(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    L_in, L_par = road_lengths(df)
    core = [ROAD_GROUPS.index(g) for g in ("I", "TOLL", "US", "ST", "FM")]
    inside = L_in[:, core, :].sum((1, 2))
    outside_par = L_par[:, core, :].sum((1, 2))
    # candidate highway inside the tract but within 15 m of its edge
    inner = sum(df.get(f"r_{g}_{b}_i15", 0) for g in ("I", "TOLL", "US", "ST", "FM")
                for b in ROAD_BUCKETS)
    inner = np.asarray(inner, dtype=float)
    total = inside + outside_par
    edge = np.divide(inner + outside_par, total + 1e-9)
    d = df.assign(hwy_km=total / 1e3, edge_share=np.where(total > 0, edge, np.nan),
                  edge_dominated=(total > 200) & (edge > 0.5))
    d = d.join(strata(d))
    has = d[d.hwy_km > 0.2]
    rows = []
    for col in ["svi_q", "tribal", "ruca", "region"]:
        t = has.groupby(col, observed=True).agg(
            tracts=("GEOID", "size"), edge_dominated_pct=("edge_dominated", "mean"),
            median_edge_share=("edge_share", "median"), pop=("pop_total", "sum"))
        t["edge_dominated_pct"] *= 100
        t.index = [f"{col}: {i}" for i in t.index]
        rows.append(t)
    tab = pd.concat(rows).round(1)
    top = (d[d.edge_dominated].sort_values("pop_total", ascending=False)
           [["GEOID", "region", "pop_total", "svi_overall", "hwy_km", "edge_share"]].head(15).round(3))
    km_edge = (inner + outside_par).sum() / 1e3
    txt = (f"Across {int((d.hwy_km > 0.2).sum())} tracts with a state/US/Interstate highway, "
           f"{km_edge:,.0f} km of highway length lies within 15 m of a tract boundary "
           f"({100 * km_edge / (total.sum() / 1e3):.1f}% of the candidate network). "
           f"{int(d.edge_dominated.sum())} tracts ({100 * d.edge_dominated.mean():.1f}% of all) "
           "have more than half of their highway on the boundary, so their road-gap value is a "
           "geometry coin flip rather than a mapping-quality signal.")
    return txt, tab, top


def machine_only_buildings(df: pd.DataFrame):
    d = df[df.b_n > 0].copy()
    d["ml_only_share"] = (d.b_ms + d.b_google) / d.b_n
    d["typed_share"] = d.b_typed / d.b_n
    d = d.join(strata(d))
    rows = []
    for col in ["svi_q", "tribal", "ruca", "region"]:
        g = d.groupby(col, observed=True)
        t = pd.DataFrame({
            "tracts": g.size(),
            "buildings": g.b_n.sum(),
            "ml_only_pct": 100 * (g.b_ms.sum() + g.b_google.sum()) / g.b_n.sum(),
            "with_use_class_pct": 100 * g.b_typed.sum() / g.b_n.sum(),
            "named_pct": 100 * g.b_named.sum() / g.b_n.sum(),
        })
        t.index = [f"{col}: {i}" for i in t.index]
        rows.append(t)
    tab = pd.concat(rows).round(1)
    top = (d[(d.b_n > 300)].sort_values(["ml_only_share", "pop_total"], ascending=[False, False])
           [["GEOID", "region", "pop_total", "tribal", "svi_overall", "b_n", "ml_only_share", "typed_share"]]
           .head(15).round(3))
    txt = (f"{100 * (d.b_ms.sum() + d.b_google.sum()) / d.b_n.sum():.1f}% of Overture buildings in the "
           "study regions exist only as ML-traced footprints (no OSM/Esri lineage). Those carry no "
           "use class, name, or height.")
    return txt, tab, top


def main():
    df = load_features()
    OUT.parent.mkdir(exist_ok=True)
    t1, tab1, top1 = edge_highways(df)
    t2, tab2, top2 = machine_only_buildings(df)
    md = ["# Bias discovery starter (challenge data only)\n",
          "Reproduce: `python src/features.py && python src/bias_discovery.py`.\n",
          "## 1. Edge highways: road gaps decided by the tract boundary\n", t1, "",
          tab1.to_markdown(), "", "Most-populated edge-dominated tracts:\n", top1.to_markdown(index=False), "",
          "## 2. Machine-only buildings: counted, but unusable for triage\n", t2, "",
          tab2.to_markdown(), "", "Large tracts with the highest machine-only share:\n",
          top2.to_markdown(index=False), ""]
    OUT.write_text("\n".join(md))
    print(OUT.read_text())


if __name__ == "__main__":
    main()
