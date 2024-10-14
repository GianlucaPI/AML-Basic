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

# Load dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="HousePrice")

# ----------------------- EDA (Exploratory Data Analysis) -----------------------

# Check for missing values
print("Missing values:\n", X.isnull().sum())

# Summary statistics
print("Summary statistics:\n", X.describe())

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

# Distribution of target variable
plt.figure(figsize=(6, 4))
sns.histplot(y, bins=50, kde=True)
plt.title("House Price Distribution")
plt.show()

# Pairplot of selected features to explore relationships
sns.pairplot(pd.concat([X[['MedInc', 'AveRooms', 'HouseAge']], y], axis=1))
plt.show()

# ----------------------- Feature Engineering -----------------------

# Create a new feature: 'RoomsPerHousehold'
X['RoomsPerHousehold'] = X['AveRooms'] / X['AveOccup']

# Log transformation to handle skewness of the target variable
y_log = np.log1p(y)  # log1p is used to handle zeros by applying log(1 + value)

# ----------------------- Data Cleaning and Preprocessing -----------------------

# Feature scaling using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)

# ----------------------- Model Training, Hyperparameter Tuning, and Evaluation -----------------------

# 1. Linear Regression (Baseline model)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
mse_linear = mean_squared_error(np.expm1(y_test), np.expm1(y_pred_linear))  # Inverse the log transformation for interpretation
print(f"Linear Regression MSE: {mse_linear}")

# 2. Ridge Regression (Hyperparameter Tuning with GridSearchCV)
ridge_model = Ridge()
ridge_params = {'alpha': [0.01, 0.1, 1.0, 10, 100]}
ridge_grid = GridSearchCV(ridge_model, ridge_params, cv=5, scoring='neg_mean_squared_error')
ridge_grid.fit(X_train, y_train)
y_pred_ridge = ridge_grid.predict(X_test)
mse_ridge = mean_squared_error(np.expm1(y_test), np.expm1(y_pred_ridge))  # Inverse the log transformation
print(f"Best Ridge Alpha: {ridge_grid.best_params_['alpha']}")
print(f"Ridge Regression MSE: {mse_ridge}")

# 3. Lasso Regression (Hyperparameter Tuning with GridSearchCV)
lasso_model = Lasso(max_iter=10000)
lasso_params = {'alpha': [0.01, 0.1, 1.0, 10, 100]}
lasso_grid = GridSearchCV(lasso_model, lasso_params, cv=5, scoring='neg_mean_squared_error')
lasso_grid.fit(X_train, y_train)
y_pred_lasso = lasso_grid.predict(X_test)
mse_lasso = mean_squared_error(np.expm1(y_test), np.expm1(y_pred_lasso))  # Inverse the log transformation
print(f"Best Lasso Alpha: {lasso_grid.best_params_['alpha']}")
print(f"Lasso Regression MSE: {mse_lasso}")

# ----------------------- Cross-Validation for Model Comparison -----------------------

# Function to display cross-validation results
def display_cv_results(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print(f"Cross-validation RMSE: {rmse_scores}")
    print(f"Mean RMSE: {rmse_scores.mean()}, Standard Deviation: {rmse_scores.std()}")

# Cross-validation for Linear Regression
print("\nLinear Regression Cross-Validation:")
display_cv_results(linear_model, X_scaled, y_log)

# Cross-validation for Ridge Regression
print("\nRidge Regression Cross-Validation:")
display_cv_results(ridge_grid.best_estimator_, X_scaled, y_log)

# Cross-validation for Lasso Regression
print("\nLasso Regression Cross-Validation:")
display_cv_results(lasso_grid.best_estimator_, X_scaled, y_log)