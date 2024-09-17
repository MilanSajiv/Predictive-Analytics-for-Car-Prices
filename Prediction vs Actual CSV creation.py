import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

# Load the dataset
df = pd.read_csv('Training_Car_Data.csv')
numerical_df = df.select_dtypes(include=['float64', 'int64'])

features = numerical_df.drop(columns=['Price'])
target = numerical_df['Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Train Linear Regression model with imputation
imputer = SimpleImputer(strategy='mean')
pipeline = make_pipeline(imputer, LinearRegression())
pipeline.fit(X_train, y_train)
lr_predictions = pipeline.predict(X_test)

# Create DataFrame with actual and predicted values
results_df = pd.DataFrame({
    'Actual': y_test.reset_index(drop=True),  # Reset index to align with predictions
    'RF_Predicted': rf_predictions,
    'LR_Predicted': lr_predictions
})

# Save to CSV
results_df.to_csv('predictions_vs_actuals.csv', index=False)