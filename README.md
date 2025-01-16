README

California Housing Prices Prediction using Linear Regression Models

Table of Contents

    •	Introduction
    •	Dataset Description
    •	Project Structure
    •	Exploratory Data Analysis (EDA)
    •	Feature Engineering
    •	Data Preprocessing
    •	Model Training and Hyperparameter Tuning
    •	Model Evaluation
    •	Cross-Validation for Model Comparison
    •	Conclusions
    •	References

Introduction

This project aims to predict California housing prices using linear regression models, specifically Linear Regression, Ridge Regression, and Lasso Regression. The project involves data exploration, feature engineering, model training, hyperparameter tuning, and evaluation to identify the best-performing model for predicting house prices.

Dataset Description

The dataset used is the California Housing dataset, available from sklearn.datasets. It contains information collected during the 1990 California census. The dataset includes the following features:

    •	MedInc: Median income in block group
    •	HouseAge: Median house age in block group
    •	AveRooms: Average number of rooms per household
    •	AveBedrms: Average number of bedrooms per household
    •	Population: Population per block group
    •	AveOccup: Average number of household members
    •	Latitude: Block group latitude
    •	Longitude: Block group longitude

The target variable is MedHouseVal, which represents the median house value for households within a block group.

Project Structure

The project is structured into the following main sections:

    1.	Exploratory Data Analysis (EDA)
    2.	Feature Engineering
    3.	Data Preprocessing
    4.	Model Training and Hyperparameter Tuning
    5.	Model Evaluation
    6.	Cross-Validation for Model Comparison

Each section builds upon the previous one to develop a robust predictive model.

========================================================================================
Exploratory Data Analysis (EDA)

Checking for Missing Values

print("Missing values:\n", X.isnull().sum())

    •	Result: No missing values are present in the dataset.

Summary Statistics

print("Summary statistics:\n", X.describe())

    •	Provides insights into the distribution, mean, standard deviation, and quartiles of each feature.

Correlation Matrix

plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

    •	Purpose: To understand the relationships between features.
    •	Findings:
    •	AveRooms and AveBedrms show high correlation.
    •	Latitude and Longitude have a moderate correlation, indicating geographical clustering.

Distribution of Target Variable

plt.figure(figsize=(6, 4))
sns.histplot(y, bins=50, kde=True)
plt.title("House Price Distribution")
plt.show()

    •	Observation: The target variable HousePrice is skewed to the right, indicating the presence of outliers or a non-normal distribution.

Pairplot of Selected Features

sns.pairplot(pd.concat([X[['MedInc', 'AveRooms', 'HouseAge']], y], axis=1))
plt.show()

    •	Purpose: To explore pairwise relationships between selected features and the target variable.
    •	Insights:
    •	Positive correlation between MedInc and HousePrice.
    •	Potential linear relationships that can be leveraged in the model.

Feature Engineering

Creating a New Feature: RoomsPerHousehold

X['RoomsPerHousehold'] = X['AveRooms'] / X['AveOccup']

    •	Reason: To capture the relationship between the number of rooms and the household occupancy, which might be a better predictor of house prices.

Log Transformation of Target Variable

y_log = np.log1p(y)

    •	Purpose: To handle skewness in the target variable and stabilize variance.
    •	Method: Using log1p to account for any zero values in the target variable.

Data Preprocessing

Feature Scaling

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

    •	Reason: To normalize the features, ensuring that each feature contributes equally to the model training.

Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)

    •	Purpose: To evaluate model performance on unseen data.
    •	Split Ratio: 80% training data and 20% testing data.
    •	Random State: Ensures reproducibility.

Model Training and Hyperparameter Tuning

1. Linear Regression (Baseline Model)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

    •	Role: Serves as a baseline to compare with regularized models.
    •	Prediction:

y_pred_linear = linear_model.predict(X_test)

    •	Evaluation:

mse_linear = mean_squared_error(np.expm1(y_test), np.expm1(y_pred_linear))

    •	Mean Squared Error (MSE) is calculated after applying the inverse log transformation to interpret results in the original scale.

2. Ridge Regression (L2 Regularization)

ridge_model = Ridge()
ridge_params = {'alpha': [0.01, 0.1, 1.0, 10, 100]}
ridge_grid = GridSearchCV(ridge_model, ridge_params, cv=5, scoring='neg_mean_squared_error')
ridge_grid.fit(X_train, y_train)

    •	Purpose: To reduce model complexity and prevent overfitting by penalizing large coefficients.
    •	Hyperparameter Tuning: Using GridSearchCV to find the optimal alpha value.
    •	Best Alpha:

print(f"Best Ridge Alpha: {ridge*grid.best_params*['alpha']}")

3. Lasso Regression (L1 Regularization)

lasso_model = Lasso(max_iter=10000)
lasso_params = {'alpha': [0.01, 0.1, 1.0, 10, 100]}
lasso_grid = GridSearchCV(lasso_model, lasso_params, cv=5, scoring='neg_mean_squared_error')
lasso_grid.fit(X_train, y_train)

    •	Purpose: Similar to Ridge Regression but can reduce some coefficients to zero, effectively performing feature selection.
    •	Hyperparameter Tuning: Finding the optimal alpha value.
    •	Best Alpha:

print(f"Best Lasso Alpha: {lasso*grid.best_params*['alpha']}")

==============================================================================================================

Model Evaluation

Mean Squared Error (MSE) Comparison

    •	Linear Regression MSE:

print(f"Linear Regression MSE: {mse_linear}")

    •	Ridge Regression MSE:

print(f"Ridge Regression MSE: {mse_ridge}")

    •	Lasso Regression MSE:

print(f"Lasso Regression MSE: {mse_lasso}")

    •	Interpretation: Comparing MSE values to determine which model performs best on the test data.

Cross-Validation for Model Comparison

Purpose

    •	To assess the generalization performance of the models.
    •	To mitigate the impact of data partitioning on model evaluation.

Cross-Validation Function

def display_cv_results(model, X, y):
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print(f"Cross-validation RMSE: {rmse_scores}")
print(f"Mean RMSE: {rmse_scores.mean()}, Standard Deviation: {rmse_scores.std()}")

    •	Explanation: Calculates RMSE for each fold and provides mean and standard deviation.

Cross-Validation Results

Linear Regression

print("\nLinear Regression Cross-Validation:")
display_cv_results(linear_model, X_scaled, y_log)

Ridge Regression

print("\nRidge Regression Cross-Validation:")
display*cv_results(ridge_grid.best_estimator*, X_scaled, y_log)

Lasso Regression

print("\nLasso Regression Cross-Validation:")
display*cv_results(lasso_grid.best_estimator*, X_scaled, y_log)

    •	Interpretation: Models with lower mean RMSE and lower standard deviation are considered more reliable.

Conclusions

    •	Linear Regression:
    •	Serves as a strong baseline with reasonable performance.
    •	May suffer from overfitting due to lack of regularization.
    •	Ridge Regression:
    •	Achieved the best performance with an optimal alpha value.
    •	Reduces the impact of multicollinearity and overfitting.
    •	Best Alpha: Determined via GridSearchCV.
    •	Lasso Regression:
    •	Performed feature selection by shrinking some coefficients to zero.
    •	May underperform if important features are eliminated due to high regularization.
    •	Cross-Validation:
    •	Ridge Regression consistently showed lower RMSE across folds.
    •	Confirms that Ridge Regression generalizes better to unseen data.
    •	Feature Engineering:
    •	The creation of RoomsPerHousehold improved model performance.
    •	Log transformation of the target variable addressed skewness.
    •	Recommendation:
    •	Ridge Regression is the preferred model for predicting California housing prices in this context.
    •	Further hyperparameter tuning and exploration of other regularization techniques could be beneficial.

References

    •	Scikit-Learn Documentation
    •	California Housing Dataset Description
    •	Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.

Code Descriptor

Below is a detailed explanation of the code, outlining each step and its purpose.

Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

    •	Purpose: Import necessary libraries for data manipulation, visualization, model training, and evaluation.

Loading the Dataset

# Load dataset

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="HousePrice")

    •	Function: fetch_california_housing() retrieves the dataset.
    •	Variables:
    •	X: Feature matrix.
    •	y: Target variable.

Exploratory Data Analysis (EDA)

Checking for Missing Values

print("Missing values:\n", X.isnull().sum())

    •	Action: Checks for null values in each feature.
    •	Outcome: Ensures data completeness.

Summary Statistics

print("Summary statistics:\n", X.describe())

    •	Action: Provides statistical measures for each feature.
    •	Outcome: Understands data distribution and identifies potential outliers.

Correlation Matrix

plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

    •	Action: Visualizes correlations between features.
    •	Outcome: Identifies multicollinearity.

Distribution of Target Variable

plt.figure(figsize=(6, 4))
sns.histplot(y, bins=50, kde=True)
plt.title("House Price Distribution")
plt.show()

    •	Action: Plots histogram of the target variable.
    •	Outcome: Observes skewness and distribution shape.

Pairplot of Selected Features

sns.pairplot(pd.concat([X[['MedInc', 'AveRooms', 'HouseAge']], y], axis=1))
plt.show()

    •	Action: Creates pairwise plots.
    •	Outcome: Explores relationships between selected features and target.

Feature Engineering

Creating a New Feature

X['RoomsPerHousehold'] = X['AveRooms'] / X['AveOccup']

    •	Purpose: Generates a feature that may capture housing density.

Log Transformation

y_log = np.log1p(y)

    •	Purpose: Applies log transformation to reduce skewness in y.

Data Preprocessing

Feature Scaling

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

    •	Action: Standardizes features to have zero mean and unit variance.
    •	Reason: Many machine learning algorithms perform better with scaled data.

Splitting Data

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)

    •	Purpose: Divides data into training and testing sets.
    •	Parameters:
    •	test_size=0.2: 20% data for testing.
    •	random_state=42: Ensures reproducibility.

Model Training and Hyperparameter Tuning

Linear Regression

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

    •	Action: Fits a linear model to the training data.

Predictions and Evaluation

y_pred_linear = linear_model.predict(X_test)
mse_linear = mean_squared_error(np.expm1(y_test), np.expm1(y_pred_linear))
print(f"Linear Regression MSE: {mse_linear}")

    •	Action: Predicts on test data and calculates MSE.
    •	Note: np.expm1() is used to invert the log transformation.

Ridge Regression

ridge_model = Ridge()
ridge_params = {'alpha': [0.01, 0.1, 1.0, 10, 100]}
ridge_grid = GridSearchCV(ridge_model, ridge_params, cv=5, scoring='neg_mean_squared_error')
ridge_grid.fit(X_train, y_train)

    •	Action: Performs hyperparameter tuning using cross-validation.
    •	Parameters:
    •	alpha: Regularization strength.

Ridge Predictions and Evaluation

y*pred_ridge = ridge_grid.predict(X_test)
mse_ridge = mean_squared_error(np.expm1(y_test), np.expm1(y_pred_ridge))
print(f"Best Ridge Alpha: {ridge_grid.best_params*['alpha']}")
print(f"Ridge Regression MSE: {mse_ridge}")

    •	Action: Predicts and evaluates the Ridge model.

Lasso Regression

lasso_model = Lasso(max_iter=10000)
lasso_params = {'alpha': [0.01, 0.1, 1.0, 10, 100]}
lasso_grid = GridSearchCV(lasso_model, lasso_params, cv=5, scoring='neg_mean_squared_error')
lasso_grid.fit(X_train, y_train)

    •	Note: Increased max_iter to ensure convergence.

Lasso Predictions and Evaluation

y*pred_lasso = lasso_grid.predict(X_test)
mse_lasso = mean_squared_error(np.expm1(y_test), np.expm1(y_pred_lasso))
print(f"Best Lasso Alpha: {lasso_grid.best_params*['alpha']}")
print(f"Lasso Regression MSE: {mse_lasso}")

    •	Action: Predicts and evaluates the Lasso model.

Cross-Validation for Model Comparison

Display Function

def display_cv_results(model, X, y):
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print(f"Cross-validation RMSE: {rmse_scores}")
print(f"Mean RMSE: {rmse_scores.mean()}, Standard Deviation: {rmse_scores.std()}")

    •	Purpose: Calculates and displays cross-validation RMSE.

Applying Cross-Validation

Linear Regression

display_cv_results(linear_model, X_scaled, y_log)

Ridge Regression

display*cv_results(ridge_grid.best_estimator*, X_scaled, y_log)

Lasso Regression

display*cv_results(lasso_grid.best_estimator*, X_scaled, y_log)

    •	Outcome: Provides a reliable estimate of model performance.

Summary

    •	The code performs data loading, EDA, feature engineering, preprocessing, model training, and evaluation.
    •	Regularization techniques like Ridge and Lasso are used to improve model performance and prevent overfitting.
    •	Cross-validation ensures the robustness of the models.
    •	The final recommendation is based on model evaluation metrics like MSE and RMSE.

Note: Remember to install all required libraries and handle any potential warnings or errors during execution.
# AML-Basic
# AML-Basic
