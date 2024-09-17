# Predictive Analytics for Car Prices

## Overview

This project involves predicting car prices using various machine learning models. The dataset contains features of cars, such as specifications and attributes, to train predictive models. The primary goal is to build, evaluate, and compare different models to predict car prices accurately.

## Table of Contents

- [Project Description](#project-description)
- [Data](#data)
- [Models](#models)
- [Evaluation](#evaluation)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Description

The Car Price Prediction project utilizes machine learning techniques to predict car prices based on various features. The project includes data preprocessing, model training, evaluation, and visualization of results. Key models used in this project include Linear Regression and Random Forest Regressor.

## Data

- **Dataset**: The dataset used is `Training_Car_Data.csv` which contains numerical features and the target variable (car price).
- **Features**: Various numerical features related to car specifications.
- **Target**: Car price.

## Models

1. **Linear Regression**: Trained using a pipeline that includes imputation for missing values.
2. **Random Forest Regressor**: Trained and tuned using GridSearchCV to find the best hyperparameters.

### Hyperparameter Tuning

- **Parameters Tuned**:
  - `n_estimators`
  - `max_depth`
  - `min_samples_split`

## Evaluation

- **Metrics**:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R-squared (R²)
- **Cross-Validation**: Performed to assess the model’s performance.

## Results

- **Feature Importance**: Visualized the top features affecting car prices.
- **Actual vs. Predicted**: Generated CSV file (`predictions_vs_actuals.csv`) for comparison and visualization in Power BI.
- **Residual Analysis**: Plotted residuals vs. predicted values to assess model performance.

## Installation

To run this project, you need Python with the following libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

You can install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
