# Credit Card Fraud Detection:

## 📌 Overview
Fraud detection is a classic "needle in a haystack" problem. In this project, we analyze 284,807 European credit card transactions where only **492 (0.172%)** are fraudulent. 

The core of this project is a comparative study between traditional **SMOTE** and the combined **SMOTETomek** (SMOTE oversampling + Tomek Links undersampling) technique to determine which better prepares highly imbalanced data for machine learning classifiers.

---

## 🧠 Key Concepts

### What is SMOTE?
**Synthetic Minority Over-sampling Technique (SMOTE)** works by selecting a minority class instance and finding its $k$-nearest neighbors. It then creates new synthetic instances by interpolating (drawing a straight line) between the instance and its neighbors.
* **Limitation:** It assumes a linear relationship between points, which can lead to "noise" if the class boundaries are complex.

### What is SMOTETomek?
**SMOTETomek** combines two resampling strategies into a single pipeline:
* **SMOTE** first oversamples the minority (fraud) class by synthesizing new points between neighboring minority instances.
* **Tomek Links** then cleans the result by removing majority-class samples that sit right on the decision boundary next to minority samples.
* **Why use it?** The combination both balances the class distribution and sharpens the boundary between classes, which can yield better recall/AUC than SMOTE alone.

---

## 🛠️ Methodology

### 1. Data Preprocessing & PCA
All features (except Time and Amount) are results of a **Principal Component Analysis (PCA)** transformation. 
* **PCA** reduces dimensionality by transforming correlated features into a set of linearly uncorrelated variables (Principal Components). This removes noise and speeds up training while retaining maximum variance.

### 2. Handling Imbalance: The Comparison
We split the pipeline into two distinct paths to compare performance:
* **Path A:** Raw (imbalanced) data.
* **Path B:** SMOTE-resampled data.
* **Path C:** SMOTETomek-resampled data (SMOTE + Tomek Links).

### 3. Classifiers
A wide range of models are trained and compared, including:

| Model | Notes |
| :--- | :--- |
| **Logistic Regression** | Baseline linear classifier. |
| **Random Forest** | Handles non-linear data via ensembles; tuned manually (`max_depth`, `n_estimators`, etc.). |
| **XGBoost** | Gradient boosting; also evaluated with manually-set hyperparameters. |
| **SVC / MLP** | Kernel-based and neural-network approaches for pattern recognition. |
| **Stacking Classifier** | Combines ~25 base estimators (AdaBoost, Bagging, Naive Bayes variants, tree-based and linear models, SVC, XGBoost, CatBoost, etc.) with an XGBoost meta-learner. |

> Note: parameter grids for `GridSearchCV` are defined in the notebook but the search itself is not run — models are instead trained with manually-chosen hyperparameters.

### 4. Evaluation Metric
Because missing a fraudulent transaction is far costlier than a false alarm, models are scored with **Accuracy, Precision, Recall, F-beta (β = 5, weighting recall heavily) and ROC-AUC** rather than accuracy alone (which is misleading on a 99.83%/0.17% class split).

---

## 📊 Experimental Setup

```python
# Conceptual Pipeline
1. Standardize Features (Time, Amount)
2. Apply PCA (Dimensionality Reduction)
3. Resampling:
    - Option 1: None (raw imbalanced data)
    - Option 2: SMOTE
    - Option 3: SMOTETomek (SMOTE + Tomek Links)
4. Train classifiers (Logistic Regression, Random Forest, XGBoost, SVC, MLP, Stacking, ...)
5. Evaluation (Accuracy, Precision, Recall, F-beta, ROC-AUC)
```

---

## 📈 Results (highlights)

Resampling clearly improves fraud detection over training on raw imbalanced data. Some of the best results from the notebook's `scores` table:

| Model | Data | Recall | AUC |
| :--- | :--- | :--- | :--- |
| Random Forest (tuned) | SMOTETomek | ~0.82 | ~0.91 |
| XGBoost | SMOTETomek | ~0.80 | ~0.90 |
| Random Forest | SMOTETomek | ~0.80 | ~0.90 |
| XGBoost | Raw data | ~0.74 | ~0.87 |

By contrast, models trained on raw imbalanced data can report >99.9% accuracy while still missing most fraud cases (e.g., one Logistic Regression run achieved ~4% recall despite ~99.8% accuracy) — illustrating why accuracy alone is a poor metric for this problem, and why resampling matters.
