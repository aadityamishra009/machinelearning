import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
file_path = r"C:\Users\Aaditya\Desktop\ML\Churn Modeling.csv"
df = pd.read_csv(file_path)

# Display basic information about the dataset
print("Dataset Information:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Define independent (X) and dependent (Y) variables
X = df[['EstimatedSalary']]
Y = df['Balance']

# Visualizing the data to identify pattern
plt.figure(figsize=(8,6))
plt.scatter(X, Y, color='blue', alpha=0.5)
plt.xlabel('Estimated Salary')
plt.ylabel('Balance')
plt.title('Estimated Salary vs Balance')
plt.text(X.max()*0.6, Y.max()*0.8, "Scatter plot shows the correlation\n between Estimated Salary and Balance.", fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
plt.show()

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Display the shape of training and testing sets
print("Training set shape:", X_train.shape, Y_train.shape)
print("Testing set shape:", X_test.shape, Y_test.shape)

# Train the Linear Regression Model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Retrieve the intercept and slope
intercept = regressor.intercept_
slope = regressor.coef_[0]
print("Intercept (b0):", intercept)
print("Slope (b1):", slope)

# Predict values
Y_pred = regressor.predict(X_test)

# Compare actual vs predicted values
comparison = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})
print("\nComparison of Actual vs Predicted values:")
print(comparison.head())

# Evaluate the model
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)

print("\nModel Evaluation Metrics:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2)

# Visualization of the regression line
plt.figure(figsize=(8,6))
plt.scatter(X_test, Y_test, color='blue', alpha=0.5, label='Actual Data')
plt.plot(X_test, Y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Estimated Salary')
plt.ylabel('Balance')
plt.title('Linear Regression - Estimated Salary vs Balance')
plt.text(X_test.max()*0.6, Y_test.max()*0.8, "Regression line represents model predictions.\n Closer alignment to data points suggests better fit.", fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
plt.legend()
plt.show()

# Observations:
# - The regression line attempts to predict the balance based on estimated salary.
# - The model’s predictive performance is evaluated using MAE, MSE, RMSE, and R2-score.
# - A high MAE or RMSE indicates larger prediction errors, meaning the model might not be very reliable.
# - If the R-squared value is low, it suggests that balance is influenced by other factors not included in this model.
# - The model may be improved by including additional relevant features or using a more complex regression approach.
