# 🏠 house-price-prediction
Predicting home sales prices in Ames, Iowa using EDA, Feature Engineering, and Ridge Regression.

![Kaggle Score](https://img.shields.io/badge/Kaggle_RMSE_Score-0.13763-blue)
![Python](https://img.shields.io/badge/Python-3.10-green)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-Ridge_Regression-orange)

---

## 📌 Executive Summary
* **Objective:** Predict home sale prices (`SalePrice`) using 80 explanatory features.
* **Evaluation Metric:** Root Mean Squared Error (RMSE) on the log-transformed price scale.
* **Key Result:** Achieved a validation log-RMSE score of **0.13763** on Kaggle using a Ridge Regression baseline model.

---

## 🛠️ Data Pipeline & Technical Approach

### 1. Exploratory Data Analysis (EDA)
* Analyzed the distribution of `SalePrice`, identifying significant right skewness ($1.88$).
* Mapped top features correlated with price, highlighting `OverallQual` ($0.79$) and `GrLivArea` ($0.71$) as primary price drivers.

### 2. Missing Value Imputation
* Categorical attributes missing values (e.g., `PoolQC`, `GarageType`, `FireplaceQu`) were imputed as `'None'` to signify the absence of features.
* Numerical attributes missing values were zero-imputed.
* `LotFrontage` was imputed using the median value per `Neighborhood`.

### 3. Feature Engineering
* **Aggregated Size Metrics:** Created `TotalSF` (Basement + 1st Floor + 2nd Floor square footage), `TotalBathrooms`, and `TotalPorchSF`.
* **Age Features:** Engineered `HouseAge` and `RemodAge` based on sale date.
* **Target Normalization:** Applied log transformation (`np.log1p`) to `SalePrice` to ensure normally distributed errors for linear modeling.

### 4. Modeling & Validation
* Encoded categorical features via **One-Hot Encoding** (`pd.get_dummies`).
* Split data into 80/20 train/validation sets.
* Trained a **Ridge Regression** model ($\alpha=10.0$) with $L_2$ regularization to handle multi-collinearity and prevent overfitting.

---

## 📁 Repository Structure
```text
├── house-price-prediction-eda-modeling.ipynb   # Main Jupyter Notebook
├── README.md                                   # Project Overview & Documentation
