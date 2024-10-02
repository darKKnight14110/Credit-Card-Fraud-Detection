# Credit Card Fraud Detection

## Overview
Credit card companies must be able to recognize fraudulent transactions to protect customers from unauthorized charges. This project aims to classify a highly unbalanced dataset using multiple classification methods to identify the best model for future test data. The dataset consists of credit card transactions made by European cardholders in September 2013, covering two days. Out of 284,807 transactions, 492 are labeled as fraud, making up just 0.172% of the dataset.

## Dataset
- **Source**: European cardholders' credit card transactions in September 2013. (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Duration**: 2 days of transaction data.
- **Transactions**: 284,807 total transactions.
- **Fraudulent transactions**: 492 (0.172%).

## Objective
The objective of this analysis is to accurately classify fraudulent credit card transactions using several classification models. Since the dataset is heavily imbalanced, the F1-Score is used as the primary evaluation metric to measure the performance of each model, focusing on balancing precision and recall.

## Methodology

### 1. **Data Preprocessing**
   - The dataset contains numerical features that were standardized.
   - No missing data was present.
   - **Principal Component Analysis (PCA)** was applied for dimensionality reduction, helping to eliminate noise and reduce computation time while retaining the most important variance from the data.

### 2. **Handling Class Imbalance**
   - The dataset is highly skewed, with only 0.172% fraud cases.
   - To address this imbalance, **Synthetic Minority Oversampling Technique (SMOTE)** was used to generate synthetic samples of the minority class (fraud cases), creating a more balanced dataset for training the models.
   - SMOTE oversamples the minority class by creating synthetic examples rather than simply duplicating existing ones.

### 3. **Feature Reduction**
   - **PCA** was employed to reduce dimensionality and capture the most significant features. By keeping the most important principal components, we reduced noise and computational overhead.

### 4. **Model Selection and Tuning**
   - Multiple machine learning models were trained to classify the transactions:
     - **Logistic Regression**: A linear model commonly used for binary classification tasks.
     - **Random Forest**: An ensemble learning method that combines multiple decision trees to improve classification accuracy.
     - **Support Vector Classifier (SVC)**: A classifier that aims to find the optimal hyperplane to separate the classes.
     - **XGBoost**: A gradient boosting model known for its speed and performance in classification tasks.
     - **MLP Classifier (Multi-Layer Perceptron)**: A neural network model with one or more hidden layers.
     - **Stacking Classifier**: A meta-learner that combines the predictions of multiple classifiers to improve performance.
   
   - **GridSearchCV** was used for hyperparameter tuning in **Random Forest** and **XGBoost** to find the optimal parameters for the models, ensuring that each model was fine-tuned for the best performance.
   
### 5. **Model Evaluation**
   - Since the dataset was highly imbalanced, **F1-Score** was chosen as the main evaluation metric, which considers both precision and recall to give a balanced measure of model performance, especially in cases of class imbalance.
   - The models were trained on the balanced data (using SMOTE) and then evaluated on the test set.

## Models Used
1. Logistic Regression
2. Random Forest (with GridSearchCV)
3. Support Vector Classifier (SVC)
4. XGBoost (with GridSearchCV)
5. MLP Classifier (Multi-Layer Perceptron)
6. Stacking Classifier

## Results
The classification models were evaluated using the F1-Score to ensure balanced performance across the highly unbalanced dataset. The final model selection was based on which classifier achieved the highest F1-Score, indicating the most suitable method for identifying fraudulent transactions.
### Best Model: Random Forest Classifier (F1-Score:0.90 & Accuracy: 99.9%+) 
