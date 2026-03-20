# The Kaggle Grandmasters' 7 Battle-Tested Techniques for Tabular Data
Source: https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/

## Foundation Principles
1. **Fast Experimentation**: GPU-accelerated dataframes (cuDF) and model training
2. **Local Validation**: k-fold CV matching data structure

## 7 Techniques

### 1. Smarter EDA
- Train vs test distribution shifts
- Temporal patterns in targets

### 2. Diverse Baselines, Built Fast
- Multiple model families simultaneously (GBTs, NNs, SVR)
- GPU enables training in minutes vs hours

### 3. Generate More Features
- Pairwise categorical interactions
- cuDF makes thousands of features practical
```python
for i,c1 in enumerate(CATS[:-1]):
    for j,c2 in enumerate(CATS[i+1:]):
        n = f"{c1}_{c2}"
        train[n] = train[c1].astype('str')+"_"+train[c2].astype('str')
```

### 4. Hill Climbing
- Start with strongest model, add others with different weights
- Keep only combinations that improve validation
- CuPy vectorizes metric calculations for thousands of weight combos in parallel
- Won Predict Calorie Expenditure (1st place): XGBoost + CatBoost + NNs + linear models

### 5. Stacking
- Train meta-model on OOF predictions of base models
- Two approaches: residuals or OOF features as inputs
- Won Podcast Listening Time (1st): 3-level stack with cuML Lasso, SVR, KNN, RF, NNs, GBTs

### 6. Pseudo-Labeling
- Use best model to infer labels on unlabeled test data
- **Use soft labels (probabilities)** not hard 0/1 — reduces noise
- Filter low-confidence samples (only add high-confidence ones)
- Avoid leakage: compute k sets of pseudo-labels with k-fold
- Won BirdCLEF 2024

### 7. Extra Training
- **Multiple random seeds**: average predictions across seeds
- **Retrain on 100% of data**: after finding hyperparams, use all training data
- Won Predicting Optimal Fertilizers: XGBoost across 100 seeds

## Our Status
| Technique | Status |
|-----------|--------|
| 1. EDA | ✓ basic |
| 2. Diverse baselines | ✓ LGB + CatBoost + MLP |
| 3. Feature engineering | ✓ 252 features |
| 4. Hill climbing | ✓ implemented |
| 5. Stacking | ✗ not done |
| 6. Pseudo-labeling | ✓ implemented |
| 7. Seeds + full retrain | ✓ 3 seeds for LGB |
