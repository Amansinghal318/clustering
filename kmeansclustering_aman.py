# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 10:35:36 2018

@author: dell 1
"""

#k means clustering
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import dataset
dataset=pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,3:5].values
#using elbow method to find optimal no of clusters

from sklearn.cluster import KMeans
wcss=[]
for i in range (1,11):
           kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=10)
           kmeans.fit(X)
           wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()
#applying k means to mall dataset
 kmeans =KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    ykmeans=kmeans.fit_predict(X)
#visualizing the clusters
plt.scatter(X(y))    
    