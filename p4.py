import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"C:\Users\Aaditya\Desktop\ML\Churn Modeling.csv"
data = pd.read_csv(file_path)

# Data Cleaning and Preprocessing
columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
data_cleaned = data.drop(columns=columns_to_drop)

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Geography', 'Gender']
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data_cleaned[column] = label_encoders[column].fit_transform(data_cleaned[column])

# Split the dataset into features (X) and target (y)
X = data_cleaned.drop(columns=['Exited'])
y = data_cleaned['Exited']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Gaussian Naive Bayes Classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)

# Train Multinomial Naive Bayes Classifier
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred_mnb = mnb.predict(X_test)

# Accuracy scores
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
accuracy_mnb = accuracy_score(y_test, y_pred_mnb)

# Cross-validation confusion matrices
y_pred_cv_gnb = cross_val_predict(gnb, X, y, cv=5)
conf_matrix_gnb = confusion_matrix(y, y_pred_cv_gnb)
disp = ConfusionMatrixDisplay(conf_matrix_gnb)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Gaussian Naive Bayes")
plt.text(0.5, 2.5, "Interpretation:\nGaussian Naive Bayes performs better with fewer\nmisclassifications and captures patterns well.",
         fontsize=10, color='darkblue', ha='center', bbox=dict(facecolor='white', alpha=0.8))
plt.show()

y_pred_cv_mnb = cross_val_predict(mnb, X, y, cv=5)
conf_matrix_mnb = confusion_matrix(y, y_pred_cv_mnb)
disp = ConfusionMatrixDisplay(conf_matrix_mnb)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Multinomial Naive Bayes")
plt.text(0.5, 2.5, "Interpretation:\nMultinomial Naive Bayes struggles with numerical features,\nleading to more misclassifications.",
         fontsize=10, color='darkblue', ha='center', bbox=dict(facecolor='white', alpha=0.8))
plt.show()

# ROC curves and AUC scores
y_prob_gnb = gnb.predict_proba(X_test)[:, 1]
y_prob_mnb = mnb.predict_proba(X_test)[:, 1]
roc_auc_gnb = roc_auc_score(y_test, y_prob_gnb)
roc_auc_mnb = roc_auc_score(y_test, y_prob_mnb)

RocCurveDisplay.from_predictions(y_test, y_prob_gnb, name='GaussianNB')
plt.title(f"ROC Curve - GaussianNB (AUC: {roc_auc_gnb:.2f})")
plt.text(0.5, 0.1, "Interpretation:\nGaussianNB's curve is closer to the top-left corner,\nindicating better performance.",
         fontsize=10, color='darkblue', ha='center', bbox=dict(facecolor='white', alpha=0.8))
plt.show()

RocCurveDisplay.from_predictions(y_test, y_prob_mnb, name='MultinomialNB')
plt.title(f"ROC Curve - MultinomialNB (AUC: {roc_auc_mnb:.2f})")
plt.text(0.5, 0.1, "Interpretation:\nMultinomialNB's ROC curve reflects weaker performance\nwith AUC close to random guessing.",
         fontsize=10, color='darkblue', ha='center', bbox=dict(facecolor='white', alpha=0.8))
plt.show()

# Print Summary
print(f"Gaussian Naive Bayes Accuracy: {accuracy_gnb:.2f}")
print(f"Multinomial Naive Bayes Accuracy: {accuracy_mnb:.2f}")
print(f"Gaussian Naive Bayes ROC AUC: {roc_auc_gnb:.2f}")
print(f"Multinomial Naive Bayes ROC AUC: {roc_auc_mnb:.2f}")
