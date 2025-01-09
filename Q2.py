# prg-2

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Load data
df = pd.read_csv('solar_efficiency_temp.csv')

# Extract required columns
X = df[['Temperature']]
y = df['Efficiency']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Plot training data
plt.scatter(X_train, y_train, color='red', label='Train Data')
plt.plot(X_train, model.predict(X_train), color='blue', label='Regression Line')
plt.xlabel('Temperature')
plt.ylabel('Efficiency')
plt.title('Training Data')
plt.legend()
plt.show()

# Plot test data
plt.scatter(X_test, y_test, color='red', label='Test Data')
plt.plot(X_test, model.predict(X_test), color='blue', label='Regression Line')
plt.xlabel('Temperature')
plt.ylabel('Efficiency')
plt.title('Test Data')
plt.legend()
plt.show()

# Perform F-test and T-test
X_with_const = sm.add_constant(X)  # Add constant for intercept
ols_model = sm.OLS(y, X_with_const).fit()

# Extract F-statistic and p-value
f_stat = ols_model.fvalue
f_p_value = ols_model.f_pvalue

# Extract t-statistic and p-value for temperature
t_stat = ols_model.tvalues['Temperature']
t_p_value = ols_model.pvalues['Temperature']

# Print results
print(f"F-statistic: {f_stat:.2f}, p-value: {f_p_value:.4f}")
print(f"t-statistic for temperature: {t_stat:.2f}, p-value: {t_p_value:.4f}")
