# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 19:35:39 2020

@author: P795864
"""

#Hierarchical Clustering

#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[: , [3,4]].values

#Using the dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendogram')
plt.xlabel('No of Customers')
plt.ylabel('Euclidean Distance')
plt.show()

#Fitting Hierarchical Clustering to the Mall Dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

#Visualising the Clusters
plt.scatter(X[y_hc == 0,0], X[y_hc == 0,1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1,0], X[y_hc == 1,1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2,0], X[y_hc == 2,1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3,0], X[y_hc == 3,1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4,0], X[y_hc == 4,1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Cluster of Customers')
plt.xlabel('Annual Income k$')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()