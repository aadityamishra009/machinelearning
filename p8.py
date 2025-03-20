import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

# Loading Dataset

df = pd.read_csv("Churn Modeling.csv")
print(df.head())

# Exploratory Data Analysis (EDA)
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df.describe())

# Dataset Visualization
plt.figure(figsize=(12, 8))
sns.boxplot(data=df.select_dtypes(include=['int64', 'float64']))
plt.title('Boxplot of all numerical features')
plt.show()

# Encoding Categorical Features
label_encoder = LabelEncoder()
df['Geography'] = label_encoder.fit_transform(df['Geography'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# Selecting relevant features
features = ['Geography', 'Gender', 'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
X = df[features].values

# Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)
scaled_df = pd.DataFrame(scaled_features, columns=features)

# Finding Optimal Number of Clusters (Elbow Method)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_df)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, 'bo-', markersize=8)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Training the K-Means Model
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(scaled_df)

# Results and Interpretation
print("Cluster Centers:")
print(kmeans.cluster_centers_)
