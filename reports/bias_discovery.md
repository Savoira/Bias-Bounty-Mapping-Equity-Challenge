# Bias discovery starter (challenge data only)

Reproduce: `python src/features.py && python src/bias_discovery.py`.

## 1. Edge highways: road gaps decided by the tract boundary

Across 6677 tracts with a state/US/Interstate highway, 47,070 km of highway length lies within 15 m of a tract boundary (28.5% of the candidate network). 3831 tracts (39.1% of all) have more than half of their highway on the boundary, so their road-gap value is a geometry coin flip rather than a mapping-quality signal.

|                          |   tracts |   edge_dominated_pct |   median_edge_share |              pop |
|:-------------------------|---------:|---------------------:|--------------------:|-----------------:|
| svi_q: SVI Q1 (least)    |     1057 |                 72.4 |                 0.7 |      4.78069e+06 |
| svi_q: SVI Q2            |     1340 |                 58.2 |                 0.5 |      5.78416e+06 |
| svi_q: SVI Q3            |     1828 |                 50.2 |                 0.5 |      7.51546e+06 |
| svi_q: SVI Q4 (most)     |     2404 |                 55.6 |                 0.5 |      9.72183e+06 |
| tribal: non-tribal       |     5936 |                 61   |                 0.5 |      2.53525e+07 |
| tribal: tribal           |      741 |                 28.5 |                 0.2 |      2.49991e+06 |
| ruca: metro              |     5051 |                 65.5 |                 0.6 |      2.22973e+07 |
| ruca: micropolitan       |      716 |                 46.9 |                 0.5 |      2.75626e+06 |
| ruca: small town         |      265 |                 32.5 |                 0.3 | 893171           |
| ruca: rural              |      645 |                 15.3 |                 0.2 |      1.90568e+06 |
| region: eastern-ok       |      912 |                 35   |                 0.3 |      3.07138e+06 |
| region: eastern-wa       |      311 |                 40.5 |                 0.3 |      1.25156e+06 |
| region: maricopa-az      |      585 |                 64.1 |                 0.5 |      2.34612e+06 |
| region: northern-ca      |      355 |                 47.9 |                 0.5 |      1.45788e+06 |
| region: south-central-tx |     4514 |                 62.9 |                 0.6 |      1.97255e+07 |

Most-populated edge-dominated tracts:

|       GEOID | region           |   pop_total |   svi_overall |   hwy_km |   edge_share |
|------------:|:-----------------|------------:|--------------:|---------:|-------------:|
| 48029172002 | south-central-tx |       30199 |         0.623 |   28.397 |        0.804 |
| 48201542902 | south-central-tx |       27220 |         0.614 |    9.549 |        1     |
| 48157673202 | south-central-tx |       24651 |         0.167 |   41.811 |        0.665 |
| 48201543007 | south-central-tx |       20530 |         0.14  |    5.286 |        1     |
| 48157673108 | south-central-tx |       18333 |         0.254 |   15.315 |        0.648 |
| 48157672901 | south-central-tx |       17341 |         0.234 |   19.768 |        0.82  |
| 48157672903 | south-central-tx |       16195 |         0.56  |    4.486 |        0.509 |
| 48029161902 | south-central-tx |       15735 |         0.794 |   29.421 |        0.572 |
| 48029172006 | south-central-tx |       15271 |         0.584 |   19.95  |        0.535 |
| 48041002100 | south-central-tx |       15243 |         0.585 |   11.166 |        0.554 |
| 48201555503 | south-central-tx |       15093 |         0.275 |   11.916 |        0.833 |
| 48157675502 | south-central-tx |       15080 |         0.28  |   15.547 |        0.631 |
| 48029182101 | south-central-tx |       15049 |         0.118 |   18.704 |        0.68  |
| 48201543004 | south-central-tx |       14783 |         0.072 |    8.146 |        1     |
| 48085030511 | south-central-tx |       14451 |         0.062 |   11.374 |        0.847 |

## 2. Machine-only buildings: counted, but unusable for triage

44.7% of Overture buildings in the study regions exist only as ML-traced footprints (no OSM/Esri lineage). Those carry no use class, name, or height.

|                          |   tracts |        buildings |   ml_only_pct |   with_use_class_pct |   named_pct |
|:-------------------------|---------:|-----------------:|--------------:|---------------------:|------------:|
| svi_q: SVI Q1 (least)    |     1900 |      3.77178e+06 |          30.2 |                 29.1 |         0.6 |
| svi_q: SVI Q2            |     1950 |      4.24128e+06 |          42   |                 19.6 |         0.8 |
| svi_q: SVI Q3            |     2497 |      5.35611e+06 |          50.9 |                 16.3 |         0.8 |
| svi_q: SVI Q4 (most)     |     3359 |      5.78342e+06 |          50.3 |                 16.7 |         1   |
| tribal: non-tribal       |     8918 |      1.71392e+07 |          40.6 |                 21.4 |         0.9 |
| tribal: tribal           |      874 |      2.03063e+06 |          78.9 |                  5.1 |         0.7 |
| ruca: metro              |     8103 |      1.45884e+07 |          36.1 |                 24.4 |         0.9 |
| ruca: micropolitan       |      759 |      1.88166e+06 |          71   |                  6.1 |         0.9 |
| ruca: small town         |      266 | 688731           |          74   |                  4.1 |         0.7 |
| ruca: rural              |      664 |      2.01095e+06 |          72.4 |                  3.8 |         0.6 |
| region: eastern-ok       |     1192 |      2.55169e+06 |          69.4 |                  5.8 |         0.7 |
| region: eastern-wa       |      415 |      1.08134e+06 |          38.1 |                 12.2 |         1.3 |
| region: maricopa-az      |     1593 |      2.90822e+06 |          15.6 |                 37.9 |         1.1 |
| region: northern-ca      |      591 |      1.16472e+06 |          50.8 |                 11.7 |         0.7 |
| region: south-central-tx |     6001 |      1.14638e+07 |          46.6 |                 19.7 |         0.8 |

Large tracts with the highest machine-only share:

|       GEOID | region           |   pop_total | tribal     |   svi_overall |   b_n |   ml_only_share |   typed_share |
|------------:|:-----------------|------------:|:-----------|--------------:|------:|----------------:|--------------:|
| 48215021906 | south-central-tx |        5598 | non-tribal |         0.728 |  1727 |               1 |             0 |
| 48215024406 | south-central-tx |        4986 | non-tribal |         0.903 |  2016 |               1 |             0 |
| 48215023525 | south-central-tx |        4390 | non-tribal |         0.866 |  1765 |               1 |             0 |
| 48215023526 | south-central-tx |        4333 | non-tribal |         0.728 |  1067 |               1 |             0 |
| 48215024404 | south-central-tx |        4148 | non-tribal |         0.845 |  2212 |               1 |             0 |
| 48215021315 | south-central-tx |        3766 | non-tribal |         0.893 |  1552 |               1 |             0 |
| 48215023520 | south-central-tx |        3757 | non-tribal |         0.885 |  1387 |               1 |             0 |
| 48215022503 | south-central-tx |        3626 | non-tribal |         0.929 |  1209 |               1 |             0 |
| 48215022011 | south-central-tx |        3585 | non-tribal |         0.906 |  2215 |               1 |             0 |
| 06067007022 | northern-ca      |        3489 | non-tribal |         0.675 |   483 |               1 |             0 |
| 48479001822 | south-central-tx |        3136 | non-tribal |         0.914 |  1297 |               1 |             0 |
| 48005000104 | south-central-tx |        2974 | non-tribal |         0.491 |  1993 |               1 |             0 |
| 48215022705 | south-central-tx |        2913 | non-tribal |         0.798 |  2183 |               1 |             0 |
| 48201322701 | south-central-tx |        2807 | non-tribal |         0.842 |   821 |               1 |             0 |
| 48167720508 | south-central-tx |        2721 | non-tribal |         0.317 |  1025 |               1 |             0 |
