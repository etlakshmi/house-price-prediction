# Machine Learning Regression Pipeline Documentation

## Overview

This document provides comprehensive documentation for a modular machine learning pipeline designed for regression tasks. The pipeline handles data preprocessing, model training, and prediction with built-in error handling and flexibility.

---

## Table of Contents

1. [Installation Requirements](#installation-requirements)
2. [Function Reference](#function-reference)
3. [Complete Workflow Example](#complete-workflow-example)
4. [Best Practices](#best-practices)
5. [Troubleshooting](#troubleshooting)

---

## Installation Requirements

```bash
pip install pandas numpy scikit-learn
```

**Required Libraries:**
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning algorithms and utilities

---

## Function Reference

### 1. `read_csv(filename)`

Reads a CSV file and returns a pandas DataFrame.

**Parameters:**
- `filename` (str): Path to the CSV file

**Returns:**
- `pd.DataFrame` or `None`: DataFrame if successful, None if error occurs

**Example:**
```python
df = read_csv("housing_data.csv")
```

---

### 2. `drop_columns(df, columns)`

Removes specified columns from the DataFrame.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `columns` (list): List of column names to drop

**Returns:**
- `pd.DataFrame`: DataFrame with specified columns removed

**Example:**
```python
df_clean = drop_columns(df, ['id', 'timestamp', 'unnecessary_col'])
```

---

### 3. `encode_categorical_columns(df, cat_cols)`

Encodes categorical columns using Label Encoding.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `cat_cols` (list): List of categorical column names to encode

**Returns:**
- `df_encoded` (pd.DataFrame): DataFrame with encoded categorical columns
- `encoders` (dict): Dictionary mapping column names to fitted LabelEncoder objects

**Example:**
```python
df_encoded, encoders = encode_categorical_columns(df, ['city', 'category', 'type'])
```

---

### 4. `fill_na_with_mean(df, cols)`

Fills missing values in specified columns with their mean values.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `cols` (list): List of column names to fill missing values

**Returns:**
- `df_filled` (pd.DataFrame): DataFrame with missing values filled
- `mean_values` (dict): Dictionary mapping column names to their mean values

**Example:**
```python
df_filled, mean_values = fill_na_with_mean(df, ['age', 'income', 'score'])
```

**Note:** Best used for numerical columns only.

---

### 5. `split_train_test(df, target, test_size=0.2, random_state=42)`

Splits data into training and testing sets.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `target` (str): Name of the target column
- `test_size` (float, optional): Proportion of data for testing (default: 0.2)
- `random_state` (int, optional): Random seed for reproducibility (default: 42)

**Returns:**
- `X_train, X_test, y_train, y_test`: Training and testing feature/target splits

**Example:**
```python
X_train, X_test, y_train, y_test = split_train_test(df, 'price', test_size=0.25)
```

---

### 6. `train_regression_models(x_train, x_test, y_train, y_test)`

Trains Linear Regression and Random Forest Regressor models and evaluates them.

**Parameters:**
- `x_train` (pd.DataFrame): Training features
- `x_test` (pd.DataFrame): Testing features
- `y_train` (pd.Series): Training target
- `y_test` (pd.Series): Testing target

**Returns:**
- `results` (dict): Dictionary containing model objects, R² scores, and RMSE values

**Example:**
```python
results = train_regression_models(X_train, X_test, y_train, y_test)

# Access results
print(f"Linear Regression R²: {results['LinearRegression']['R2']:.4f}")
print(f"Random Forest R²: {results['RandomForestRegressor']['R2']:.4f}")
```

**Models Trained:**
1. **Linear Regression** - Simple linear model
2. **Random Forest Regressor** - Ensemble model with 200 trees

---

### 7. `predict_from_user_input(model, user_input, feature_columns, encoders, mean_values)`

Makes predictions on new user input data.

**Parameters:**
- `model`: Trained model object
- `user_input` (dict): Dictionary with feature names as keys
- `feature_columns` (list/Index): Column names from training data
- `encoders` (dict): Dictionary of LabelEncoder objects
- `mean_values` (dict): Dictionary of mean values for imputation

**Returns:**
- `float`: Predicted value

**Example:**
```python
user_data = {
    'city': 'New York',
    'bedrooms': 3,
    'bathrooms': 2,
    'sqft': 1500
}

prediction = predict_from_user_input(
    model=results['RandomForestRegressor']['model'],
    user_input=user_data,
    feature_columns=X_train.columns,
    encoders=encoders,
    mean_values=mean_values
)

print(f"Predicted value: ${prediction:,.2f}")
```

---

## Complete Workflow Example

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Step 1: Load data
df = read_csv("real_estate_data.csv")

# Step 2: Drop unnecessary columns
df = drop_columns(df, ['id', 'timestamp', 'url'])

# Step 3: Handle missing values (numerical columns)
numerical_cols = ['age', 'sqft', 'price_per_sqft']
df, mean_values = fill_na_with_mean(df, numerical_cols)

# Step 4: Encode categorical columns
categorical_cols = ['neighborhood', 'property_type', 'condition']
df, encoders = encode_categorical_columns(df, categorical_cols)

# Step 5: Split data
X_train, X_test, y_train, y_test = split_train_test(df, 'price', test_size=0.2)

# Step 6: Train models
results = train_regression_models(X_train, X_test, y_train, y_test)

# Step 7: Compare models
print("\nModel Performance Comparison:")
print("-" * 50)
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"  R² Score: {metrics['R2']:.4f}")
    print(f"  RMSE: ${metrics['RMSE']:,.2f}")

# Step 8: Select best model
best_model_name = max(results, key=lambda k: results[k]['R2'])
best_model = results[best_model_name]['model']
print(f"\nBest Model: {best_model_name}")

# Step 9: Make predictions
new_property = {
    'neighborhood': 'Downtown',
    'property_type': 'Condo',
    'condition': 'Excellent',
    'age': 5,
    'sqft': 1200,
    'price_per_sqft': 250
}

predicted_price = predict_from_user_input(
    best_model,
    new_property,
    X_train.columns,
    encoders,
    mean_values
)

print(f"\nPredicted Price: ${predicted_price:,.2f}")
```

---

## Best Practices

### Data Preprocessing

1. **Handle Missing Values Before Encoding**
   - Fill numerical missing values with mean/median
   - Fill categorical missing values with mode or 'Unknown'

2. **Separate Numerical and Categorical Processing**
   ```python
   # Identify column types
   numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
   categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
   ```

3. **Check Data Quality**
   ```python
   # Check for missing values
   print(df.isnull().sum())
   
   # Check data types
   print(df.dtypes)
   
   # Check for duplicates
   print(f"Duplicates: {df.duplicated().sum()}")
   ```

### Model Training

1. **Scale Features for Linear Models**
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   ```

2. **Cross-Validation for Better Evaluation**
   ```python
   from sklearn.model_selection import cross_val_score
   scores = cross_val_score(model, X_train, y_train, cv=5, 
                            scoring='r2')
   print(f"CV R² Score: {scores.mean():.4f} (+/- {scores.std():.4f})")
   ```

3. **Hyperparameter Tuning**
   ```python
   from sklearn.model_selection import GridSearchCV
   
   param_grid = {
       'n_estimators': [100, 200, 300],
       'max_depth': [10, 20, 30, None],
       'min_samples_split': [2, 5, 10]
   }
   
   grid_search = GridSearchCV(RandomForestRegressor(random_state=42),
                              param_grid, cv=5, scoring='r2')
   grid_search.fit(X_train, y_train)
   best_model = grid_search.best_estimator_
   ```

### Prediction

1. **Validate Input Data**
   ```python
   def validate_input(user_input, expected_features):
       for feature in expected_features:
           if feature not in user_input:
               raise ValueError(f"Missing required feature: {feature}")
   ```

2. **Handle Unseen Categories**
   ```python
   # In predict_from_user_input, add:
   try:
       df[col] = le.transform(df[col].astype(str))
   except ValueError:
       # Use most frequent category or default value
       df[col] = le.transform([le.classes_[0]])[0]
   ```

---

## Troubleshooting

### Common Issues

**Issue 1: ValueError during encoding**
```
ValueError: y contains previously unseen labels
```
**Solution:** The prediction input contains a category not seen during training. Either add default handling or retrain with more diverse data.

**Issue 2: Feature mismatch**
```
ValueError: X has different number of features than during fitting
```
**Solution:** Ensure user input contains all required features. Use `df.reindex(columns=feature_columns, fill_value=0)`.

**Issue 3: Poor model performance (low R²)**
**Solutions:**
- Add more relevant features
- Try feature engineering
- Use more advanced models (Gradient Boosting, XGBoost)
- Collect more training data
- Check for data leakage

**Issue 4: High RMSE**
**Solutions:**
- Scale/normalize features
- Remove outliers
- Try log transformation on target variable
- Use ensemble methods

---

## Performance Metrics Explained

### R² Score (Coefficient of Determination)
- **Range:** -∞ to 1 (typically 0 to 1)
- **Interpretation:** Proportion of variance in target explained by features
- **Good Score:** > 0.7 is generally considered good
- **Formula:** 1 - (SS_res / SS_tot)

### RMSE (Root Mean Squared Error)
- **Range:** 0 to ∞
- **Interpretation:** Average prediction error in same units as target
- **Good Score:** Depends on scale of target variable
- **Formula:** √(Σ(y_true - y_pred)² / n)

---

## Advanced Extensions

### Add More Models

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# In train_regression_models function, add:
gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbr.fit(x_train, y_train)
y_pred_gbr = gbr.predict(x_test)
results["GradientBoosting"] = {
    "model": gbr,
    "R2": r2_score(y_test, y_pred_gbr),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_gbr))
}
```

### Feature Importance Analysis

```python
# For Random Forest or Gradient Boosting
model = results['RandomForestRegressor']['model']
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))
```

### Save and Load Models

```python
import joblib

# Save model
joblib.dump(best_model, 'best_regression_model.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(mean_values, 'mean_values.pkl')

# Load model
loaded_model = joblib.load('best_regression_model.pkl')
loaded_encoders = joblib.load('encoders.pkl')
loaded_means = joblib.load('mean_values.pkl')
```

---

## License

This pipeline is provided as-is for educational and commercial use.

## Contributing

Feel free to extend and modify this pipeline for your specific use cases.