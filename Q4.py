# prg-4

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_csv('/mnt/data/sales_synthetic.csv')

model = LinearRegression()

x1 = df[['AdvertisingExpenditure']]
y = df[['SalesRevenue']]
x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size=0.3, random_state=42)
model.fit(x_train, y_train)
print("Analysis for AdvertisingExpenditure:")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
plt.scatter(x_train, y_train, color='red', label='Training Data')
plt.scatter(x_test, y_test, color='blue', label='Testing Data')
plt.plot(x_train, model.predict(x_train), color='green', label='Regression Line')
plt.xlabel('Advertising Expenditure')
plt.ylabel('Sales Revenue')
plt.title('Advertising Expenditure vs Sales Revenue')
plt.legend()
plt.show()

x2 = df[['StoreLocation']]
x_train, x_test, y_train, y_test = train_test_split(x2, y, test_size=0.3, random_state=42)
model.fit(x_train, y_train)
print("\nAnalysis for StoreLocation:")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
plt.scatter(x_train, y_train, color='red', label='Training Data')
plt.scatter(x_test, y_test, color='blue', label='Testing Data')
plt.plot(x_train, model.predict(x_train), color='green', label='Regression Line')
plt.xlabel('Store Location')
plt.ylabel('Sales Revenue')
plt.title('Store Location vs Sales Revenue')
plt.legend()
plt.show()

x3 = df[['Competition']]
x_train, x_test, y_train, y_test = train_test_split(x3, y, test_size=0.3, random_state=42)
model.fit(x_train, y_train)
print("\nAnalysis for Competition:")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
plt.scatter(x_train, y_train, color='red', label='Training Data')
plt.scatter(x_test, y_test, color='blue', label='Testing Data')
plt.plot(x_train, model.predict(x_train), color='green', label='Regression Line')
plt.xlabel('Competition')
plt.ylabel('Sales Revenue')
plt.title('Competition vs Sales Revenue')
plt.legend()
plt.show()

X = df[['AdvertisingExpenditure', 'StoreLocation', 'Competition']]
Y = df['SalesRevenue']
X = sm.add_constant(X)
ols_model = sm.OLS(Y, X).fit()
print("\nStatistical Analysis:")
print(f"F-statistic: {ols_model.fvalue:.2f}")
print(f"t-statistic for AdvertisingExpenditure: {ols_model.tvalues['AdvertisingExpenditure']:.2f}")
print(f"t-statistic for StoreLocation: {ols_model.tvalues['StoreLocation']:.2f}")
print(f"t-statistic for Competition: {ols_model.tvalues['Competition']:.2f}")
