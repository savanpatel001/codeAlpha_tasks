import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns   

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




# Load the dataset

file_path = 'p3\\Advertising.csv'  # Update with the correct path to your dataset
data = pd.read_csv(file_path)
print(data.head(100)) # Check the first 100 rows of the dataset



# dataset information

print(data.info())
print(data.describe())


# Check for missing values

print(data.isnull().sum())



#  data visualization

plt.figure(figsize=(10, 6))
sns.scatterplot(x='TV', y='Sales', data=data)
plt.title('TV Advertising vs Sales')
plt.xlabel('TV Advertising')
plt.ylabel('Sales')
plt.show()



# feature selection 

X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']


# split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Train the linear regression model

model = LinearRegression()
model.fit(X_train, y_train)


# Make predictions on the test set

y_pred = model.predict(X_test)


# Evaluate the model

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R2:", r2)



# compare actual vs predicted values

comparison = pd.DataFrame({'Actual sales': y_test, 'Predicted sales': y_pred})
print(comparison.head(20))



# Visualize the actual vs predicted sales

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()


# predict future sales

future_sales = model.predict([[250, 40, 50]])  # Example : TV=150, Radio=30, Newspaper=20
print("Predicted Future Sales:", future_sales[0])
