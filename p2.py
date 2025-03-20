import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
file_path = "/mnt/data/linear regression (1) - linear regression (1).csv"
df = pd.read_csv(file_path)

# Display dataset information
print("Dataset Information:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Selecting features and target variable
X = df[['sqft_living']]  # Independent variable
y = df['price']  # Dependent variable

# Visualizing the relationship between sqft_living and price
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', alpha=0.5)
plt.xlabel('Living Area (sqft)')
plt.ylabel('House Price ($)')
plt.title('Living Area vs House Price')
plt.show()

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Retrieve model parameters
intercept = model.intercept_
slope = model.coef_[0]
print("Intercept (b0):", intercept)
print("Slope (b1):", slope)

# Making predictions
y_pred = model.predict(X_test)

# Compare actual vs predicted values
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nComparison of Actual vs Predicted values:")
print(comparison.head())

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2)

# Visualization of the regression line
plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Living Area (sqft)')
plt.ylabel('House Price ($)')
plt.title('Linear Regression - Living Area vs House Price')
plt.legend()
plt.show()

# Interpretation:
# - The regression line represents the model’s predictions.
# - A low R-squared value indicates that sqft_living alone is not a strong predictor of house prices.
# - The model may be improved by including additional features such as location, number of bedrooms, or condition.
# - A high MAE and RMSE suggest significant prediction errors, indicating the need for a more complex model.