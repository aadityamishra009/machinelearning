import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import warnings

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("Iris.csv", encoding='utf-8')

# Ensure column names are consistent
df.columns = df.columns.str.strip().str.lower()  # Remove spaces and convert to lowercase

# Print first few rows to verify
print(df.head())

# Display column names to check if 'species' exists
print("Column names:", df.columns)

# Dataset info
df.info()

# Shape of dataset
print(df.shape)

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values before imputation:\n", missing_values)

# Handle missing values by imputing numerical columns with mean
numerical_cols = df.select_dtypes(include=[np.number]).columns
df[numerical_cols] = df[numerical_cols].apply(lambda col: col.fillna(col.mean()))

# Verify missing values are handled
missing_values = df.isnull().sum()
print("Missing values after imputation:\n", missing_values)

# Summary statistics
print("Detailed EDA")
print("-----------------")
print("\nSummary Statistics:\n", df.describe())

# Univariate Analysis
print("Univariate Analysis")
print("---------------------")
for col in numerical_cols:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    sns.histplot(df[col], kde=True, ax=axes[0], color="blue")
    axes[0].set_title(f'Histogram of {col}')

    # Box Plot
    sns.boxplot(y=df[col], ax=axes[1], color="red")
    axes[1].set_title(f'Box Plot of {col}')

    plt.show()

# Correlation Matrix Heatmap
print("Bivariate Analysis")
plt.figure(figsize=(8, 6))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Identify the categorical column (species)
categorical_col = df.select_dtypes(include=['object']).columns[0]  # Assuming only one categorical column

# Encode categorical labels (species)
label_encoder = LabelEncoder()
df[categorical_col] = label_encoder.fit_transform(df[categorical_col])

# Splitting dataset into features and labels
X = df.drop(columns=[categorical_col])  # Features
y = df[categorical_col]  # Labels

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define MLP model
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', solver='adam', max_iter=500, random_state=42)

# Train the model
mlp.fit(X_train, y_train)

# Make predictions
y_pred = mlp.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
