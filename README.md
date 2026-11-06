# Kaggle Notebooks

A collection of Kaggle competition entries and experiments across machine learning, chemistry, survival analysis, and self-supervised learning.

---

## Competitions

### Playground Series S6E11 — Credit-card fraud
Shallow-tree stack with in-fold target encoding, decimal-digit keys and public prediction banks; public LB 0.95211.
See [s6e11-cc-fraud/](s6e11-cc-fraud/) for what worked, what did not, and the notebook.

---

### NeurIPS 2025 — Open Polymer Challenge
Predicting polymer properties from molecular structure using SMILES representations.

| Notebook | Approach | Score |
|----------|----------|-------|
| [CombiModel](combimodel-0-057-score/) | Combined model ensemble | 0.057 |
| [SMILES→RDKit→LGBM](smiles-rdkit-lgbm-ftw/) | Molecular fingerprints via RDKit + LightGBM | — |

**Key techniques:** SMILES parsing, RDKit molecular fingerprints, ChemBERT, gradient boosting, ensemble stacking

---

### CIBMTR — Equity in Post-HCT Survival Predictions
Predicting event-free survival after stem cell transplant with equity constraints across racial groups.

| Notebook | Approach |
|----------|----------|
| [XGBoost + Equity Metric](survival-predictions-xgboost/) | XGBoost with custom C-Index metric penalising racial disparity |
| [Group by Race](group-with-race/) | Race-stratified analysis and feature engineering |

**Key techniques:** Survival analysis, Kaplan-Meier transformation, custom concordance index, XGBoost  
**Full project:** [kaggle_survivor_comp](https://github.com/alexchilton/kaggle_survivor_comp)

---

### Playground Series S6E3 — Predict Customer Churn
Binary classification predicting customer churn. Metric: AUC-ROC.

| Notebook | Approach | OOF AUC | LB Score |
|----------|----------|---------|----------|
| [LightGBM Baseline](churn-s6e3/) | LightGBM 5-fold CV + feature engineering | 0.91618 | TBD |

**Key techniques:** LightGBM, stratified K-fold, target encoding, pairwise interaction features, digit decomposition of numerics, original IBM Telco dataset stats as features
**Competition:** [playground-series-s6e3](https://www.kaggle.com/competitions/playground-series-s6e3)

---

### Playground Series S6E4 — Predict Irrigation Need
Three-class classification of irrigation need (Low / Medium / High). Metric: balanced accuracy.
The target is heavily imbalanced — `High` is roughly 3% of the training rows.

| Notebook | Approach | OOF Balanced Accuracy | LB Score |
|----------|----------|-----------------------|----------|
| [3-Model Blend + Bias Tune](s6e4-irrigation/s6e4-pairwise-combos-3-model-blend-bias-tune.ipynb) | XGBoost + CatBoost + LightGBM weighted blend, then per-class bias tuning | 0.97896 | TBD |
| [LightGBM Baseline](s6e4-irrigation/irrigation_need.ipynb) | LightGBM 5-fold CV + EDA and feature engineering | 0.96520 | TBD |

Individual OOF before blending: CatBoost 0.97746, XGBoost 0.97736, LightGBM 0.97649.
Blend weights XGB 0.15 / CB 0.60 / LGB 0.25 gave 0.97745; per-class bias tuning took it to 0.97896.

**Key techniques:** Pairwise interaction features, stratified K-fold, weighted 3-model blending, per-class bias tuning for a balanced-accuracy metric on an imbalanced target
**Competition:** [playground-series-s6e4](https://www.kaggle.com/competitions/playground-series-s6e4)

---

### House Prices — Advanced Regression
Predicting residential property sale prices.

| Notebook | Approach | Votes | Rank |
|----------|----------|-------|------|
| [Ensemble Regression](housing-price-regression-removed-columns/) | Random Forest + XGBoost + LightGBM + CatBoost stacking | 4 | 363 |

**Key techniques:** Ensemble methods, stacking, GridSearchCV hyperparameter tuning, feature engineering

---

### Titanic — Machine Learning from Disaster
Binary survival classification.

| Notebook | Approach | Votes |
|----------|----------|-------|
| [SimCLR Titanic](titanic-comp-simclr/) | Self-supervised contrastive learning (SimCLR) | 1 |
| [Titanic Baseline](titanic/) | Logistic Regression, SVM, Random Forest, KNN, Decision Trees | 1 |

**Notable:** SimCLR applied to tabular data — an unusual approach using contrastive representation learning on a classic classification problem.

---

## Experiments & Tutorials

| Notebook | Description | Votes |
|----------|-------------|-------|
| [YData Profiling EDA](ydata-profiling-tutorial-quick-efficient-eda/) | Automated exploratory data analysis with YData Profiling | 4 |
| [PINN Playing Around](pinn-playing-around/) | Physics-Informed Neural Networks — experiments with PDE-constrained learning | — |
| [GardenWise Fine-Tuning](fine-tuning-ai-gardenwise/) | Kaggle version of TinyLlama LoRA fine-tuning on gardening Q&A | — |

---

## Related Projects

- [Kaggle Survival Competition (full repo)](https://github.com/alexchilton/kaggle_survivor_comp) — complete codebase for the HCT survival analysis competition
- [TinyLlama Fine-Tuning](https://github.com/alexchilton/fine_tuning_tiny_llama_with_garden_questions) — full LoRA fine-tuning project behind the GardenWise notebook
