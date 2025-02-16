# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 20:41:06 2020

@author: P795864
"""

#K-Means Cluster

#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[: , [3,4]].values


#Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, max_iter = 300, n_init = 10, init = 'k-means++', random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) 
plt.plot(range(1,11), wcss)
plt.title('The ELbow Method')
plt.xlabel('No of Clusters')
plt.ylabel('WCSS')
plt.show()    
    
    
#Applying K-Means to the Dataset
 kmeans = KMeans(n_clusters = 5, max_iter = 300, n_init = 10, init = 'k-means++', random_state = 0)
 y_kmeans = kmeans.fit_predict(X)
  
#Visualising the Clusters
plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2,1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3,0], X[y_kmeans == 3,1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4,0], X[y_kmeans == 4,1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of Clients')
plt.xlabel('Annual Salary k$')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()



