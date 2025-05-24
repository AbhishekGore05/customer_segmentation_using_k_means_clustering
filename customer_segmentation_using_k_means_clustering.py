
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

"""Data Collection & Analysis"""

# loading the data from csv file
customer_data = pd.read_csv('Mall_Customers.csv')
# first 5 rows in the dataframe
customer_data.head()

# finding the number of rows and columns
customer_data.shape

# getting some informations about the dataset
customer_data.info()

# checking for missing values
customer_data.isnull().sum()

"""Choosing the Annual Income Column & Spending Score column"""

X = customer_data.iloc[:,[3,4]].values

print(X)

"""Choosing the number of clusters

WCSS  ->  Within Clusters Sum of Squares
"""

# finding wcss value for different number of clusters

wcss = []

for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
  kmeans.fit(X)

  wcss.append(kmeans.inertia_)

# Gender Plot Bar
genders = customer_data.Gender.value_counts()
plt.figure(figsize=(5, 6))
sns.barplot(x=genders.index, y=genders.values)
plt.xlabel('Gender')
plt.ylabel('count')
plt.title('Gender Distribution')
plt.show()

# Age Bar Plot
plt.figure(figsize=(10, 6))
sns.histplot(customer_data['Age'], bins=20, kde=True)
plt.title('Age Distribution of Customers')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# plot the centroids
plt.figure(figsize=(10, 6))
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')
plt.title('Customer Groups')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()


# plot an elbow graph
plt.figure(figsize=(10, 6))
sns.set()
plt.plot(range(1,11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Optimum Number of Clusters = 5
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)

# label for each data point based on their cluster
Y = kmeans.fit_predict(X)

print('Label for each data Point:-')
print(Y)
"""5 Clusters -  0, 1, 2, 3, 4

Visualizing all the Clusters
"""

# plotting all the clusters and their Centroids

plt.figure(figsize=(10,6))
plt.scatter(X[Y==0,0], X[Y==0,1], s=50, c='green', label='Cluster 1')
plt.scatter(X[Y==1,0], X[Y==1,1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y==2,0], X[Y==2,1], s=50, c='yellow', label='Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s=50, c='violet', label='Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s=50, c='blue', label='Cluster 5')

# plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='cyan', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()



