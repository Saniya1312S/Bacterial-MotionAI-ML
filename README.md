# 🦠 BacMotionAI - Classifying Bacterial Chemical Conditions from Motion

[![tests](https://github.com/Saniya1312S/Bacterial-MotionAI-ML/actions/workflows/ci.yml/badge.svg)](https://github.com/Saniya1312S/Bacterial-MotionAI-ML/actions)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)

**[Live Dashboard](https://Saniya1312S.github.io/Bacterial-MotionAI-ML/)** | interactive science dashboard with model results, trajectory explorer, and evaluation plots.


An end-to-end machine-learning project: **given the (x, y) path of one swimming bacterium, predict which chemical
condition it is in** - `none`, `attractant`, `repellent` or `antibiotic`.

The data is simulated, so I control the ground truth. The focus of the project is the **ML workflow**:
feature engineering, leakage-safe preprocessing, baselines, cross-validated tuning, honest evaluation and error analysis.

---

## 1. Results at a glance

| | |
|:--|:--|
| Task | 4-class classification (balanced, 600 samples per class) |
| Data | 2,400 simulated trajectories, 11 features, split 80% train / 20% test |
| Best model (chosen by cross-validation) | Random Forest |
| **Test accuracy** | **87.3%** (95% CI 84.2% - 90.2%), macro-F1 0.873 |
| Chance level / simple baseline | 25% / 80.8% (Logistic Regression on speed features only) |

### Baselines (test set)

| Baseline | Accuracy | Macro-F1 |
|:--|:-:|:-:|
| Majority class | 25.0% | 0.100 |
| Decision tree (depth 3), all features | 78.8% | 0.789 |
| Decision tree (depth 3), `mean_speed` only | 78.8% | 0.789 |
| Logistic Regression, speed features only (3) | 80.8% | 0.809 |

### Tuned models

| Model | CV macro-F1 | Train acc | Test acc (95% CI) | Test macro-F1 | ROC-AUC |
|:--|:-:|:-:|:-:|:-:|:-:|
| Random Forest | 0.884 | 100.0% | 87.3% (84.2% - 90.2%) | 0.873 | 0.977 |
| Gradient Boosting | 0.880 | 97.4% | 87.9% (84.6% - 90.8%) | 0.879 | 0.978 |
| Logistic Regression | 0.854 | 86.2% | 85.8% (82.7% - 89.2%) | 0.858 | 0.968 |
| KNN | 0.781 | 84.2% | 77.9% (74.0% - 81.5%) | 0.779 | 0.926 |

### Per-class results (Random Forest)

| Class | Precision | Recall | F1 |
|:--|:-:|:-:|:-:|
| none | 0.75 | 0.78 | 0.77 |
| attractant | 0.92 | 0.88 | 0.90 |
| repellent | 0.85 | 0.83 | 0.84 |
| antibiotic | 0.98 | 1.00 | 0.99 |

![Model comparison](images/model_comparison.png)

### What the results actually say

- **Speed does most of the work.** A depth-3 tree on `mean_speed` alone reaches 78.8%. The extra features
  (turning angle, reorientation count, persistence) add roughly 5-7 points over speed-only models.
- **Random Forest, Gradient Boosting and Logistic Regression are statistically tied.** Their confidence
  intervals overlap heavily, so I do not claim one is "the best". The Random Forest was selected because it
  had the highest *cross-validation* score (0.884 vs 0.880), not the highest test score.
- **The Random Forest overfits** (100% train vs 87% test). The test score still matches the CV score, so the
  estimate is trustworthy, but a shallower forest or regularisation would be a sensible next step.
- **Most errors are `none` vs `repellent`**, because their speeds overlap. `antibiotic` is almost perfect (it is much slower).
- **Narrow channels are hardest** (77% accuracy at 10 µm vs 92-95% at 50-75 µm): constant wall collisions disturb the path.

| Confusion matrix | Permutation importance |
|:-:|:-:|
| ![cm](images/confusion_matrix.png) | ![imp](images/feature_importance.png) |

---

## 2. ML pipeline

```
 simulate trajectories  ->  extract 11 features  ->  EDA  ->  split 80/20
        |                                                         |
 data/*.csv (2,400 rows)                         5-fold CV + GridSearch (train set only)
                                                                  |
                              pick best model by CV  ->  evaluate ONCE on the test set
                                                                  |
                                      CI, per-class metrics, error analysis, feature importance
```

| Step | Script | Output |
|:--|:--|:--|
| 1. Generate data | `scripts/generate_dataset.py` | `data/bacterial_motility_dataset.csv` |
| 2. EDA | `scripts/eda.py` | `images/eda_*.png` |
| 3. Train + tune | `scripts/train.py` | `models/*.joblib`, `results/cv_results.csv` |
| 4. Evaluate | `scripts/evaluate.py` | `results/*.csv`, `results/results.md`, `images/*.png` |
| 5. Demo | `streamlit run streamlit_app.py` | interactive app with the real model |

### Data and features

Each bacterium is simulated for 6 seconds (run-and-tumble swimming in a narrow channel). Only the **positions**
are used, as a microscope tracker would provide. The 11 features:

| Group | Features |
|:--|:--|
| Speed | `mean_speed`, `std_speed`, `cv_speed` |
| Path shape | `straightness`, `mean_turn_angle`, `turn_bias`, `persistence_1s`, `reorient_count` |
| Wall contact | `mean_alignment`, `near_wall_frac`, `wall_contact_frac` |

Channel width and the wall-interaction setting are saved as metadata for error analysis but are **never model inputs**.

### Good practice used

- **No label circularity:** the target is the chemical used in the simulation, not a rule computed from the features.
- **No preprocessing leakage:** `StandardScaler` is inside a scikit-learn `Pipeline`, so it is fitted inside each CV fold.
- **Test set used once:** hyper-parameters and the final model are chosen by cross-validation on the training set only.
- **Baselines and confidence intervals:** every result is compared with simple baselines and shown with a 95% bootstrap CI.
- **Class imbalance is not an issue** (perfectly balanced), so no SMOTE is needed. Macro-F1 is still reported.
- **Tests and CI:** 10 pytest tests (simulation, features, end-to-end pipeline) run on every push.

---

## 3. Quick start

```bash
git clone https://github.com/Saniya1312S/Bacterial-MotionAI-ML.git
cd Bacterial-MotionAI-ML
pip install -r requirements.txt

python scripts/generate_dataset.py   # ~15 seconds
python scripts/eda.py
python scripts/train.py              # a few minutes (grid search)
python scripts/evaluate.py
python -m pytest tests -q

streamlit run streamlit_app.py       # demo app
```

The repository already contains the generated dataset and the trained `best_model.joblib`, so the app works right
after cloning. For the notebook: `pip install jupyter`, then open `notebooks/01_walkthrough.ipynb`.

## 4. Project structure

```
├── data/                  dataset (CSV) + description
├── models/                best_model.joblib
├── results/               metrics tables (CSV / Markdown)
├── images/                EDA + evaluation plots
├── notebooks/01_walkthrough.ipynb   short executed walkthrough
├── scripts/               generate_dataset, eda, train, evaluate, export_dashboard_data
├── src/                   simulation, features, data, models, inference
├── tests/                 pytest tests
├── site/                  static dashboard (HTML/CSS/JS + exported JSON)
└── streamlit_app.py       demo app
```

## 5. Limitations (please read)

- **The data is synthetic.** The ~87% accuracy shows how separable my *simulated* conditions are. It is not a
  claim about real bacteria; the class overlap is decided by the parameters I chose in `src/simulation.py`.
- Simulation parameters (speed, turning noise, drift) are simplified and not fitted to experimental data.
- A single train/test split of 480 test samples gives a confidence interval of about ±3%.
- Hyper-parameter grids are small, and some best values sit at the edge of the grid.

## 6. Ideas for further work

- Add channel width as an input feature and measure the gain.
- Try real bacterial tracking data to test the sim-to-real gap.
- Add probability calibration and SHAP explanations.
- Reduce Random Forest overfitting (limit depth, more regularisation).

## 7. Tech stack

Python · NumPy · pandas · scikit-learn · Matplotlib · seaborn · Streamlit · pytest · GitHub Actions


## 8. Interactive Dashboard

The site/ directory contains a static science dashboard deployed to GitHub Pages:

- **Dataset Explorer** - class distributions, feature statistics, provenance
- **Model Lab** - interactive metric selector, confusion matrix, per-class results
- **Trajectory Explorer** - precomputed trajectories with the trained model's actual predictions
- **Model Insights** - evaluation plots generated by the scripts (not fabricated)
- **Pipeline Diagram** - links to every script and module

The dashboard loads exported JSON from site/data/ and existing plots from site/images/.
No metrics are hard-coded in the frontend.

### Regenerate dashboard data

`ash
python scripts/export_dashboard_data.py
`

This reads data/, 
esults/, models/best_model.joblib, and writes JSON to site/data/.
It does not modify the dataset, model, or results.

### Local testing

`ash
cd site
python -m http.server 8000
# Open http://localhost:8000
`

## 9. GitHub Pages deployment

The dashboard is deployed automatically via .github/workflows/deploy-pages.yml
when changes to site/ are pushed to main.

**Setup (one-time):** Repository Settings > Pages > Build and deployment > Source > **GitHub Actions**

Live URL: https://Saniya1312S.github.io/Bacterial-MotionAI-ML/

## License

MIT
