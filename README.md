# Credit Card Fraud Detection: A2 Copula vs. SMOTE

## 📌 Overview
Fraud detection is a classic "needle in a haystack" problem. In this project, we analyze 284,807 European credit card transactions where only **492 (0.172%)** are fraudulent. 

The core of this project is a comparative study between traditional **SMOTE** and the advanced **A2 Copula-based oversampling** technique to determine which better prepares highly imbalanced data for machine learning classifiers.

---

## 🧠 Key Concepts

### What is SMOTE?
**Synthetic Minority Over-sampling Technique (SMOTE)** works by selecting a minority class instance and finding its $k$-nearest neighbors. It then creates new synthetic instances by interpolating (drawing a straight line) between the instance and its neighbors.
* **Limitation:** It assumes a linear relationship between points, which can lead to "noise" if the class boundaries are complex.

### What is A2 Copula-based Oversampling?
**Copulas** are mathematical functions used to describe the dependence structure between random variables, independent of their marginal distributions. 
* **A2 Copula** (Archimedean Copula) specifically excels at capturing non-linear and asymmetric dependencies. 
* **Why use it?** Unlike SMOTE, A2 Copula oversampling models the *joint distribution* of the minority class. This allows it to generate synthetic data that respects the underlying statistical "shape" of fraud, leading to more realistic samples.

---

## 🛠️ Methodology

### 1. Data Preprocessing & PCA
All features (except Time and Amount) are results of a **Principal Component Analysis (PCA)** transformation. 
* **PCA** reduces dimensionality by transforming correlated features into a set of linearly uncorrelated variables (Principal Components). This removes noise and speeds up training while retaining maximum variance.

### 2. Handling Imbalance: The Comparison
We split the pipeline into two distinct paths to compare performance:
* **Path A:** Traditional SMOTE.
* **Path B:** A2 Copula Oversampling.

### 3. Classifiers & Hyperparameter Tuning
We utilize **GridSearchCV** for exhaustive search over specified parameter values to optimize the following models:

| Model | Tuning Strategy | Why use it? |
| :--- | :--- | :--- |
| **Logistic Regression** | Regularization (C) | Baseline linear classifier. |
| **Random Forest** | `n_estimators`, `max_depth` | Handles non-linear data via ensembles. |
| **XGBoost** | `learning_rate`, `subsample` | State-of-the-art gradient boosting. |
| **SVC** | Kernel type, Gamma | Finds optimal hyperplanes in high-dim space. |
| **MLP** | Hidden layer sizes | Neural network approach for pattern recognition. |
| **Stacking** | Meta-learner | Combines predictions of the above to reduce bias/variance. |

---

## 📊 Experimental Setup

```python
# Conceptual Pipeline
1. Standardize Features
2. Apply PCA (Dimensionality Reduction)
3. Oversampling:
    - Option 1: SMOTE
    - Option 2: A2 Copula
4. Hyperparameter Tuning (GridSearchCV)
5. Evaluation (F1-Score, Precision-Recall Curve)
