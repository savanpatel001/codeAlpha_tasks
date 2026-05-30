import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns   
import numpy as np  



# Load the dataset

file_path = 'p1\\Unemployment in India.csv'  # Update this path to the location of your dataset
df = pd.read_csv(file_path)

print(df.head(50))  # Display the first few rows of the dataset




# Understand the Dataset

  
print(df.info())  # check dataset information

print(df.shape)    # check shape

print(df.columns)  # check column names 




# Data Cleaning
 
 
df.columns = df.columns.str.strip()  # Remove extra spaces from column names

print(df.columns)  # Check the cleaned column names 

 
print(df.isnull().sum())  # Check for missing values in each column

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)  # Convert 'Date' column to datetime format, specifying dayfirst=True if the date format is day/month/year

print(df['Date'].dtype)  # Check the data type of 'Date' column after conversion




# Basic Data Exploration

print(df.describe ())  # statistical summary

print(df['Region'].unique())  # unique Regions

avg_unemployment = df['Estimated Unemployment Rate (%)'].mean()
print("Average Unemployment Rate:", avg_unemployment)




# Data Visualization

plt.figure(figsize=(10,5))
plt.hist(df['Estimated Unemployment Rate (%)'], bins=20)
plt.xlabel('Estimated Unemployment Rate (%)')
plt.ylabel('Frequency')
plt.title('Distribution of Unemployment Rates')
plt.show()


# Mothly Trend visualization

plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Estimated Unemployment Rate (%)'])
plt.xlabel('Date')
plt.ylabel('Estimated Unemployment Rate (%)')
plt.title('Unemployment Rate Over Time')
plt.xticks(rotation=45)
plt.show()




# Analyze Covid-19 Impact

pre_covid = df[df['Date'] < '2020-03-01'] # Pre-COVID period
post_covid = df[df['Date'] >= '2020-03-01'] # Post-COVID period

pre_covid_avg = pre_covid['Estimated Unemployment Rate (%)'].mean()  # Calculate average unemployment rate for pre-COVID period
post_covid_avg = post_covid['Estimated Unemployment Rate (%)'].mean() # Calculate average unemployment rate for post-COVID period  

comparison = pd.DataFrame({
    'Period': ['Pre-COVID', 'Post-COVID'],
    'Average Unemployment Rate (%)': [pre_covid_avg, post_covid_avg]
})
comparison.plot(x='Period', y='Average Unemployment Rate (%)', kind='bar', figsize=(7,5))
plt.title('Average Unemployment Rate Before and After COVID-19')
plt.ylabel('Average Unemployment Rate (%)')
plt.show()




# state wise analysis 

state_unemployment = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False)
print(state_unemployment.head(10))  # Display the average unemployment rate by state, sorted in descending order

plt.figure(figsize=(12,6))
state_unemployment.head(10).plot(kind='bar')
plt.title('Top 10 States by Average Unemployment Rate') 
plt.xlabel('State')
plt.ylabel('Average Unemployment Rate (%)')  
plt.xticks(rotation=45)  
plt.show()




# Seasonal Trend Analysis

df['Month'] = df['Date'].dt.month  # Extract month from 'Date' column
monthly_trend = df.groupby('Month')['Estimated Unemployment Rate (%)'].mean()  #Calculate average unemployment rate for each month
print(monthly_trend)  

plt.figure(figsize=(10,5))
monthly_trend.plot(kind='line', marker='o')
plt.title('Average Unemployment Rate by Month')
plt.xlabel('Month')
plt.ylabel('Average Unemployment Rate (%)')
plt.grid(True)
plt.show()




# Correlation Analysis

correlation_matrix = df.corr(numeric_only=True)  # Calculate correlation matrix
print(correlation_matrix)  

plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('correlation heatmap')
plt.show()