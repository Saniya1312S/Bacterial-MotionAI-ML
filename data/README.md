# Dataset

`bacterial_motility_dataset.csv` is **synthetic**: it is created by `scripts/generate_dataset.py`
(2,400 rows, fixed seeds, fully reproducible).

| Column | Meaning |
|:--|:--|
| 11 feature columns | See `src/features.py` (speed, path shape, wall contact) |
| `chemical` | **Target**: `none`, `attractant`, `repellent`, `antibiotic` (600 rows each) |
| `channel_width` | Simulation setting in µm (metadata, *not* a model input) |
| `hydro` | Wall hydrodynamic interaction on/off (metadata, *not* a model input) |
| `seed` | Random seed of that trajectory |

Design: 6 channel widths x 4 chemicals x 2 wall settings x 50 bacteria = 2,400 rows, perfectly balanced.
