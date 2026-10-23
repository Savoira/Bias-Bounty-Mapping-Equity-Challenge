"""Write a Zindi submission from the Monte Carlo model.

    python src/make_submission.py                       # default priors, full columns
    python src/make_submission.py --priors priors/x.json --draws 600 --out submissions/x.csv
    python src/make_submission.py --columns minimal     # GEOID + coverage_gap_score only
    python src/make_submission.py --no-usfs-buildings   # strict mode, see README

Component columns are written as E[gap * 1{defined}], which is what the organiser CSV
stores (0 when undefined), and the *_defined flags as P(defined) > 0.5.
"""
from __future__ import annotations

import argparse

import pandas as pd

from config import OUT_DIR, ROOT
from model import load_features, load_priors, simulate

COMPONENTS = [  # submission column -> model key
    ("transport", "road"), ("building", "bld"), ("poi", "poi"),
    ("poi_gap_fire", "fire"), ("poi_gap_ems", "ems"), ("poi_gap_schools", "school"),
    ("poi_gap_cbp", "cbp"),
]


def to_submission(pred: pd.DataFrame, template: pd.DataFrame, columns: str = "full") -> pd.DataFrame:
    p = pred.set_index("GEOID")
    sub = pd.DataFrame({"GEOID": template.GEOID})
    sub["coverage_gap_score"] = p.loc[sub.GEOID, "coverage_gap_score"].to_numpy()
    if columns == "minimal":
        return sub
    sub["region"] = p.loc[sub.GEOID, "region"].to_numpy()
    for col, key in COMPONENTS:
        gap = (p[f"E_{key}"] * p[f"P_{key}"]).loc[sub.GEOID].to_numpy()
        dfl = (p[f"P_{key}"] > 0.5).loc[sub.GEOID].to_numpy()
        if col.startswith("poi_gap_"):
            suffix = col.removeprefix("poi_gap_")
            sub[col], sub[f"poi_defined_{suffix}"] = gap, dfl
        else:
            sub[f"{col}_gap"], sub[f"{col}_defined"] = gap, dfl
    return sub[template.columns]


def validate(sub: pd.DataFrame, template: pd.DataFrame) -> None:
    assert len(sub) == len(template), "row count differs from sample submission"
    assert (sub.GEOID.to_numpy() == template.GEOID.to_numpy()).all(), "GEOID order differs"
    assert sub.GEOID.str.len().eq(11).all(), "GEOID lost leading zeros"
    assert not sub.isna().any().any(), f"blank cells in {sub.columns[sub.isna().any()].tolist()}"
    s = sub.coverage_gap_score
    assert s.between(0, 1).all(), "scores outside [0, 1]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--priors", default=None, help="JSON overriding Priors fields")
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-usfs-buildings", action="store_true")
    ap.add_argument("--columns", choices=["full", "minimal"], default="full")
    ap.add_argument("--out", default=str(OUT_DIR / "submission.csv"))
    ap.add_argument("--save-pred", default=None, help="also save the raw model frame (parquet)")
    a = ap.parse_args()

    template = pd.read_csv(ROOT / "SampleSubmission.csv", dtype={"GEOID": str})
    df = load_features()
    missing = set(template.GEOID) - set(df.GEOID)
    assert not missing, f"{len(missing)} sample GEOIDs have no features; run features.py"
    pred = simulate(df, load_priors(a.priors), n_draws=a.draws, seed=a.seed,
                    use_usfs_buildings=not a.no_usfs_buildings)
    sub = to_submission(pred, template, a.columns)
    validate(sub, template)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sub.to_csv(a.out, index=False)
    if a.save_pred:
        pred.to_parquet(a.save_pred, index=False)
    print(f"wrote {a.out}  rows={len(sub)}  mean={sub.coverage_gap_score.mean():.4f}")
    print(sub.groupby(pred.set_index("GEOID").loc[sub.GEOID, "region"].to_numpy())
          .coverage_gap_score.describe().round(3))


if __name__ == "__main__":
    main()
