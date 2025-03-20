import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder

# Load dataset
file_path = r"C:\Users\Aaditya\Desktop\ML\Churn Modeling.csv"
data = pd.read_csv(file_path)

# EDA
print("Dataset Overview:")
print(data.info())
print("\nDescriptive Statistics:")
print(data.describe())

# Handle categorical variables
le = LabelEncoder()
data['Geography'] = le.fit_transform(data['Geography'])
data['Gender'] = le.fit_transform(data['Gender'])

# Correlation heatmap (drop non-numeric columns)
numeric_data = data.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Split the dataset into features (X) and target (y)
X = data.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
y = data['Exited']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision Tree with Gini Index
clf_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
clf_gini.fit(X_train, y_train)
y_pred_gini = clf_gini.predict(X_test)

# Decision Tree with Entropy
clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf_entropy.fit(X_train, y_train)
y_pred_entropy = clf_entropy.predict(X_test)

# Accuracy Scores
print("\nAccuracy with Gini Index:", accuracy_score(y_test, y_pred_gini))
print("Accuracy with Entropy:", accuracy_score(y_test, y_pred_entropy))

# Pre-Pruning with Gini Index and Entropy
pre_pruned_gini = DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_split=10, min_samples_leaf=5, random_state=42)
pre_pruned_gini.fit(X_train, y_train)
y_pred_pre_pruned_gini = pre_pruned_gini.predict(X_test)

pre_pruned_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_split=10, min_samples_leaf=5, random_state=42)
pre_pruned_entropy.fit(X_train, y_train)
y_pred_pre_pruned_entropy = pre_pruned_entropy.predict(X_test)

print("\nAccuracy with Pre-Pruned Gini:", accuracy_score(y_test, y_pred_pre_pruned_gini))
print("Accuracy with Pre-Pruned Entropy:", accuracy_score(y_test, y_pred_pre_pruned_entropy))

# Visualizing Pre-Pruned Trees
plt.figure(figsize=(15, 10))
plot_tree(pre_pruned_gini, feature_names=X.columns, class_names=['Not Exited', 'Exited'], filled=True)
plt.title("Pre-Pruned Decision Tree (Gini Index)")
plt.show()

plt.figure(figsize=(15, 10))
plot_tree(pre_pruned_entropy, feature_names=X.columns, class_names=['Not Exited', 'Exited'], filled=True)
plt.title("Pre-Pruned Decision Tree (Entropy)")
plt.show()

# Post-Pruning with Cost Complexity Pruning (ccp_alpha)
# Get the effective alphas for pruning
path_gini = clf_gini.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas_gini = path_gini.ccp_alphas

path_entropy = clf_entropy.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas_entropy = path_entropy.ccp_alphas

# Train pruned trees with different alpha values
pruned_trees_gini = [DecisionTreeClassifier(criterion='gini', random_state=42, ccp_alpha=alpha).fit(X_train, y_train) for alpha in ccp_alphas_gini]
pruned_trees_entropy = [DecisionTreeClassifier(criterion='entropy', random_state=42, ccp_alpha=alpha).fit(X_train, y_train) for alpha in ccp_alphas_entropy]

# Evaluate post-pruned trees for Gini
acc_pruned_gini = [accuracy_score(y_test, tree.predict(X_test)) for tree in pruned_trees_gini]

# Evaluate post-pruned trees for Entropy
acc_pruned_entropy = [accuracy_score(y_test, tree.predict(X_test)) for tree in pruned_trees_entropy]

# Plot the effect of pruning on accuracy
plt.figure(figsize=(12, 6))
plt.plot(ccp_alphas_gini, acc_pruned_gini, marker='o', label='Gini Post-Pruned Accuracy')
plt.plot(ccp_alphas_entropy, acc_pruned_entropy, marker='o', label='Entropy Post-Pruned Accuracy')
plt.title("Effect of Post-Pruning on Accuracy")
plt.xlabel("Alpha (ccp_alpha)")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

# Selecting the best pruned trees
best_alpha_gini = ccp_alphas_gini[np.argmax(acc_pruned_gini)]
best_alpha_entropy = ccp_alphas_entropy[np.argmax(acc_pruned_entropy)]

print("\nBest Alpha for Gini:", best_alpha_gini)
print("Best Alpha for Entropy:", best_alpha_entropy)

# Retrain using the best alpha
post_pruned_gini = DecisionTreeClassifier(criterion='gini', random_state=42, ccp_alpha=best_alpha_gini)
post_pruned_gini.fit(X_train, y_train)
y_pred_post_pruned_gini = post_pruned_gini.predict(X_test)

post_pruned_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42, ccp_alpha=best_alpha_entropy)
post_pruned_entropy.fit(X_train, y_train)
y_pred_post_pruned_entropy = post_pruned_entropy.predict(X_test)

print("\nAccuracy with Post-Pruned Gini:", accuracy_score(y_test, y_pred_post_pruned_gini))
print("Accuracy with Post-Pruned Entropy:", accuracy_score(y_test, y_pred_post_pruned_entropy))

# Visualizing Post-Pruned Trees
plt.figure(figsize=(15, 10))
plot_tree(post_pruned_gini, feature_names=X.columns, class_names=['Not Exited', 'Exited'], filled=True)
plt.title("Post-Pruned Decision Tree (Gini Index)")
plt.show()

plt.figure(figsize=(15, 10))
plot_tree(post_pruned_entropy, feature_names=X.columns, class_names=['Not Exited', 'Exited'], filled=True)
plt.title("Post-Pruned Decision Tree (Entropy)")
plt.show()
