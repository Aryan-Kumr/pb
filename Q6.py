# program-6

from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import pyplot as plt

# Load the dataset
df = pd.read_csv("income_clustering.csv")
print(df.head())

# Scatter plot of Age vs Income
plt.scatter(df.Age, df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.title('Age vs Income')
plt.show()

# KMeans clustering with 3 clusters
km = KMeans(n_clusters=3, random_state=42)
y_predicted = km.fit_predict(df[['Age', 'Income($)']])
df['cluster'] = y_predicted
print(df.head())

# Separate the data into clusters
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

# Scatter plot with clusters
plt.scatter(df1.Age, df1['Income($)'], color='green', label='Cluster 0')
plt.scatter(df2.Age, df2['Income($)'], color='red', label='Cluster 1')
plt.scatter(df3.Age, df3['Income($)'], color='blue', label='Cluster 2')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='purple', marker='*', label='Centroid')
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend()
plt.title('KMeans Clustering')
plt.show()

# Finding the optimal number of clusters using Elbow Method
sse = []
k_rng = range(1, 10)
for k in k_rng:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(df[['Age', 'Income($)']])
    sse.append(km.inertia_)

# Plot SSE vs K
plt.plot(k_rng, sse, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method for Optimal K')
plt.show()
