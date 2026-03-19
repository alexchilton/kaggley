# Kaggle Notebooks

A collection of Kaggle competition entries and experiments across machine learning, chemistry, survival analysis, and self-supervised learning.

---

## Competitions

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
