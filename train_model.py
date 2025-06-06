import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_selection import SelectFromModel
import joblib

# Load enriched data
df = pd.read_csv("enriched_leads.csv")

# Feature list and target
features = [
    'Industry', 'Product/service', 'Business type',
    'Employee count', 'Revenue', 'Year founded',
    'BBB rating', "Owner's title", 'Source'
]
target = 'Converted'

X = df[features]
y = df[target]

# ====================
# Handle missing values in target
# ====================
X = X[y.notna()]
y = y[y.notna()]

# Identify columns
categorical_cols = ['Industry', 'Product/service', 'Business type', 'BBB rating', "Owner's title", 'Source']
numerical_cols = ['Employee count', 'Revenue', 'Year founded']

# ====================
# Log-transform skewed numerical columns
# ====================
log_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')),
    ('log', FunctionTransformer(func=np.log1p, validate=False)),
    ('scale', StandardScaler())
])

# ====================
# Categorical transformer
# ====================
categorical_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# ====================
# Full Preprocessor
# ====================
preprocessor = ColumnTransformer(transformers=[
    ('num', log_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# ====================
# Complete Pipeline with optional feature selection
# ====================
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42))
])

# ====================
# Split and Train
# ====================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model_pipeline.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model_pipeline.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"Test RMSE: {rmse:.4f}")

# Cross-validation score
cv_scores = cross_val_score(model_pipeline, X, y, cv=5, scoring='neg_root_mean_squared_error')
print(f"CV RMSE: {-np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# Save model
joblib.dump(model_pipeline, "lead_scoring_model.pkl")
print("Model saved as: lead_scoring_model.pkl")
