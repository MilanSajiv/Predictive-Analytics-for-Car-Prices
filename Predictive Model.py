import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import joblib

# Load the dataset
df = pd.read_csv('Training_Car_Data.csv')
numerical_df = df.select_dtypes(include=['float64', 'int64'])

features = numerical_df.drop(columns=['Price'])
target = numerical_df['Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Feature Importance using RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
feature_importances = pd.Series(model.feature_importances_, index=features.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.xlabel('Feature Importance')
plt.title('Top 10 Features by Importance')
plt.show()

# Train and evaluate Linear Regression model with imputation
imputer = SimpleImputer(strategy='mean')
pipeline = make_pipeline(imputer, LinearRegression())
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

def evaluate_model(y_true, y_pred, model_name):
    print(f'Evaluation for {model_name}:')
    print(f'MAE: {mean_absolute_error(y_true, y_pred)}')
    print(f'RMSE: {mean_squared_error(y_true, y_pred, squared=False)}')
    print(f'R²: {r2_score(y_true, y_pred)}')
    print('-----------------------------------')

evaluate_model(y_test, y_pred, "Linear Regression with Imputation")

# Hyperparameter tuning for RandomForestRegressor
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f'Best Parameters: {best_params}')

# Evaluate the tuned model
model = RandomForestRegressor(**best_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
evaluate_model(y_test, y_pred, "Tuned Random Forest")

# Cross-validation
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f'Cross-Validation Scores: {-scores.mean()}')

# Residuals plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.show()