# Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Display full output
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)



# Load the Dataset
car = pd.read_csv('C:\\Users\\savan\\OneDrive\\Desktop\\PROJECT\\p2\\car data.csv')
print(car.head(100)) # Display first 100 rows



#  Understand the Dataset
print(car.shape)
print(car.info())
print(car.describe())



# Check Missing Values
print(car.isnull().sum())



# Feature Engineering
car['Car_Age'] = 2025 - car['Year']
car.drop(['Year', 'Car_Name'], axis=1, inplace=True)



# Convert Categorical Data into Numerical Data
le = LabelEncoder()

car['Fuel_Type'] = le.fit_transform(car['Fuel_Type'])
car['Selling_type'] = le.fit_transform(car['Selling_type'])
car['Transmission'] = le.fit_transform(car['Transmission'])
print(car.head())



# Correlation Heatmap
plt.figure(figsize=(10,7))
sns.heatmap(car.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()



# Separate Features and Target
X = car.drop('Selling_Price', axis=1)
y = car['Selling_Price']



# Split the Dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Train Linear Regression Model

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)



# Predict Values

y_pred_lr = lr_model.predict(X_test)



#  Evaluate Linear Regression Model

mae = mean_absolute_error(y_test, y_pred_lr)
mse = mean_squared_error(y_test, y_pred_lr)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_lr)

print('MAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)
print('R2 Score:', r2)



#  Train Random Forest Regressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)



# Predict Using Random Forest

y_pred_rf = rf_model.predict(X_test)



# Evaluate Random Forest Model

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print('Random Forest MAE:', mae_rf)
print('Random Forest RMSE:', rmse_rf)
print('Random Forest R2 Score:', r2_rf)



# Compare Actual vs Predicted Values

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_rf)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Car Prices')
plt.show()



#  Feature Importance

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})

importance = importance.sort_values(by='Importance', ascending=False)

print(importance)


