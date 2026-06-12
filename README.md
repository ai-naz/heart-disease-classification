# Heart Disease Classification

A machine learning project to predict heart disease using the UCI Heart Disease dataset.

## Overview

This project applies binary classification to predict whether a patient has heart disease based on clinical features such as age, cholesterol, chest pain type, and maximum heart rate.

## Dataset

- **Source**: UCI Machine Learning Repository — Heart Disease Dataset (id=45)
- **Size**: 303 patients, 14 features
- **Target**: Binary (0 = no disease, 1 = disease present)

## Project Structure
notebooks/
├── eda.ipynb           # Exploratory Data Analysis & Preprocessing
└── modelling.ipynb     # Model Training & Evaluation

## Methods

- Missing value imputation (median)
- Feature scaling (StandardScaler)
- Binary target transformation

## Models

| Model               | Accuracy | Precision | Recall | F1-score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.885    | 0.879     | 0.906  | 0.892    |
| KNN (k=5)           | 0.918    | 0.935     | 0.906  | 0.921    |
| SVM (RBF kernel)    | 0.902    | 0.933     | 0.875  | 0.903    |

**Best model: KNN** — highest accuracy and F1-score on test set.

## Key Findings

- KNN achieved the best test performance but showed higher variance in cross-validation
- Logistic Regression and SVM were more stable across folds
- Chest pain type (cp) and maximum heart rate (thalach) were the strongest predictors

## Requirements
pandas
numpy
matplotlib
seaborn
scikit-learn
ucimlrepo

## How to Run

1. Clone the repository
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn ucimlrepo`
3. Open notebooks in Jupyter and run all cells
