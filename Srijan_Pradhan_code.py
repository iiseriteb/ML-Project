# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 21:45:02 2023

@author: pradh
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import chi2, SelectKBest, f_regression, RFE, RFECV
from sklearn.feature_selection import SequentialFeatureSelector, SelectFromModel, VarianceThreshold, mutual_info_regression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Loading Data
training_data = pd.read_csv(r'C:\Users\pradh\OneDrive\Desktop\ML Project\training_data.csv')
training_targets = pd.read_csv(r'C:\Users\pradh\OneDrive\Desktop\ML Project\training_data_targets.csv', header=None)
test_data = pd.read_csv(r"C:\Users\pradh\OneDrive\Desktop\ML Project\test_data.csv")
training_targets.columns = ['Price']
car = pd.concat([training_data, training_targets], axis=1)

# Distribution of Price
plt.figure(figsize=(10, 6))
sns.kdeplot(data=car, x='Price', fill=True, label='Actual Prices')
plt.title('Density Graph for Actual Prices in Training Data')
plt.xlabel('Price')
plt.ylabel('Density')
plt.legend()
plt.show()

# Data Preprocessing
car['Power'] = car['Power'].replace('null bhp', np.nan)
test_data['Power'] = test_data['Power'].replace('null bhp', np.nan)

car['Mileage'] = car['Mileage'].str.replace(' kmpl', '').str.replace(' km/kg', '').astype(float)
car['Engine'] = car['Engine'].str.replace(' CC', '').astype(float)
car['Power'] = car['Power'].str.replace(' bhp', '').astype(float)

test_data['Mileage'] = test_data['Mileage'].str.replace(' kmpl', '').str.replace(' km/kg', '').astype(float)
test_data['Engine'] = test_data['Engine'].str.replace(' CC', '').astype(float)
test_data['Power'] = test_data['Power'].str.replace(' bhp', '').astype(float)

imputer = SimpleImputer(strategy='median')
car[['Power', 'Engine']] = imputer.fit_transform(car[['Power', 'Engine']])
test_data[['Power', 'Engine']] = imputer.transform(test_data[['Power', 'Engine']])

car = car.dropna(subset=['Seats', 'Mileage'])
test_data = test_data.dropna(subset=['Seats', 'Mileage'])

# Ohe Encoder
car_ohe = car.copy()
car_ohe = car_ohe.drop(columns=['Brand', 'Year', 'Location'])
test_ohe = test_data.copy()
test_ohe = test_ohe.drop(columns=['Brand', 'Year', 'Location'])
car_ohe = pd.get_dummies(car_ohe, columns=['Fuel_Type', 'Transmission', 'Owner_Type'],drop_first=True)
test_ohe = pd.get_dummies(test_ohe, columns=['Fuel_Type', 'Transmission', 'Owner_Type'],drop_first=True)

X_ohe = car_ohe.drop(columns=['Price'])
y_ohe = car_ohe['Price']

X_ohe_train, X_ohe_test, y_ohe_train, y_ohe_test = train_test_split(X_ohe, y_ohe, test_size=0.2, random_state=42)

# Feature Selection
# Variance Threshold
variance_selector = VarianceThreshold(threshold=0.01)
X_ohe_train_variance_selected = variance_selector.fit_transform(X_ohe_train)
selected_features_variance = X_ohe_train.columns[variance_selector.get_support()]
print("Selected features using Variance Threshold:")
print(selected_features_variance)

# Correlation Analysis
correlation_matrix_ohe = car_ohe.corr()
correlation_with_price_ohe = correlation_matrix_ohe['Price'].sort_values(ascending=False)
selected_features_corr_ohe = correlation_with_price_ohe[correlation_with_price_ohe.abs() > 0.1].index.tolist()
print("Selected features using Correlation Analysis:")
print(selected_features_corr_ohe)

# Univariate Feature Selection (Select K Best)
skbest = SelectKBest(score_func=f_regression, k=5)
skbest.fit(X_ohe_train, y_ohe_train)
selected_features_univariate_ohe = X_ohe_train.columns[skbest.get_support()]
print(selected_features_univariate_ohe)

# Mutual Information
mi_ohe = mutual_info_regression(X_ohe_train, y_ohe_train)
selected_features_mi_ohe = X_ohe_train.columns[mi_ohe > 0.01]
print(selected_features_mi_ohe)

# Sequential Feature Selection
model_rf = RandomForestRegressor()
sfs = SequentialFeatureSelector(model_rf, n_features_to_select='auto', direction='forward', scoring=None, cv=5)
sfs.fit(X_ohe_train, y_ohe_train)
selected_features_sfs_ohe = list(X_ohe.columns[sfs.get_support()])
print(selected_features_sfs_ohe)

# Selected the Feature of Sequential Feature Selection
selected_features_ohe = selected_features_sfs_ohe

preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), selected_features_ohe)
    ],
    remainder='passthrough'
)



models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'SVR': SVR(),
    'Random Forest': RandomForestRegressor(max_features=None),
    'Decision Tree': DecisionTreeRegressor(random_state=40),
    'Polynomial Regression': make_pipeline(StandardScaler(), PolynomialFeatures(), Ridge()),
    'AdaBoost': AdaBoostRegressor(base_estimator=DecisionTreeRegressor(), random_state=42)
}

param_grid = {
    'Linear Regression': {},
    'Ridge': {'ridge__alpha': [0.01, 0.1, 1], 'ridge__solver': ['auto', 'svd', 'lsqr']},
    'Lasso': {'lasso__alpha': [0.1, 1, 5], 'lasso__selection': ['cyclic', 'random']},
    'SVR': {'svr__C': [1, 10,100], 'svr__epsilon': [10,0.1, 0.01], 'svr__kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
    'Random Forest': {
    'randomforestregressor__n_estimators': [50, 100, 200],
    'randomforestregressor__criterion': ['mae', 'friedman_mse', 'poisson'],
    'randomforestregressor__max_depth': [10, 20, 30],
    'randomforestregressor__max_features': [None, 'sqrt', 'log2']
},
    'Decision Tree': {
    'decisiontreeregressor__max_depth': [None, 10, 20,40,45,60],
    'decisiontreeregressor__criterion': ['mae', 'friedman_mse', 'poisson'],
    'decisiontreeregressor__max_features': [None, 'sqrt', 'log2'],
    'decisiontreeregressor__ccp_alpha': [0.009, 0.01, 0.05, 0.1,0.5]
},

    'Polynomial Regression': {
    'pipeline__polynomialfeatures__degree': [2, 3],
    'pipeline__ridge__alpha': [0.1, 1, 10],
    'pipeline__ridge__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'saga']
},

    'AdaBoost': {
    'adaboostregressor__n_estimators': [50, 100, 200,400],
    'adaboostregressor__learning_rate': [0.01, 0.1, 1,10],
    'adaboostregressor__base_estimator': [LinearRegression(),DecisionTreeRegressor(random_state=0),Ridge(alpha=1.0,solver='lbfgs',positive=True)],
    'adaboostregressor__random_state': [0, 10]
}
}



best_models = {}

for name, model in models.items():
    if name in param_grid:
        pipeline = make_pipeline(preprocessor, model)
        grid_search = GridSearchCV(pipeline, param_grid[name], scoring='neg_mean_squared_error',cv=5)
        grid_search.fit(X_ohe, y_ohe)  

        best_models[name] = {
            'best_estimator': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }

# Best Parameters
for name, details in best_models.items():
    print(f"Best model for {name}: {details['best_estimator']}")
    print(f"Best parameters: {details['best_params']}")
    print(f"Best negative mean squared error: {details['best_score']}\n")


best_model_estimate_name = None
best_r2_score = -float('inf')

for name, details in best_models.items():
    best_model_estimate = details['best_estimator']

    y_ohe_pred = best_model_estimate.predict(X_ohe_test)

    r2 = r2_score(y_ohe_test, y_ohe_pred)
    mse = mean_squared_error(y_ohe_test, y_ohe_pred)
    rmse = mean_squared_error(y_ohe_test, y_ohe_pred, squared=False)
    mae = mean_absolute_error(y_ohe_test, y_ohe_pred)

    print(f"Metrics for {name}:")
    print(f"R-squared: {r2}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}\n")

    # Updating the best model if the current model has a higher R2 score
    if r2 > best_r2_score:
        best_r2_score = r2
        best_model_estimate_name = name

print(f"The best model based on R-squared is {best_model_estimate_name} with R-squared: {best_r2_score:.4f}")

# Best Model
best_estimator = Pipeline(steps=[
    ('columntransformer', ColumnTransformer(
        remainder='passthrough',
        transformers=[('scaler', StandardScaler(), selected_features_ohe)])),
    ('adaboostregressor', AdaBoostRegressor(
        base_estimator=DecisionTreeRegressor(random_state=0),
        learning_rate=0.1, n_estimators=400, random_state=0))
])

# Fitting the best model on the entire dataset
best_estimator.fit(X_ohe[selected_features_ohe], y_ohe)

#Checking Performance on overall set to check overfitting
y_ohe_pred = best_estimator.predict(X_ohe[selected_features_ohe])
r2 = r2_score(y_ohe, y_ohe_pred)
print(r2)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_ohe, y_ohe_pred, color='blue', alpha=0.5)
plt.plot(y_ohe, y_ohe,  '--', color='red', linewidth=2)
plt.title('Scatter Plot of Predicted vs Actual Prices on Test Data')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

# Predictions on the test set
test_predictions = best_estimator.predict(test_ohe[selected_features_ohe])

# Save predictions to a file
np.savetxt('Srijan_Pradhan_Output.txt', test_predictions, fmt='%f')
