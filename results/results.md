### Baselines (test set)

| Baseline | Accuracy | Macro-F1 |
|:--|:-:|:-:|
| Majority class | 25.0% | 0.100 |
| Decision tree (depth 3), all features | 78.8% | 0.789 |
| Decision tree (depth 3), mean_speed only | 78.8% | 0.789 |
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
