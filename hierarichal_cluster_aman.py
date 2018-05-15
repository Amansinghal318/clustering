# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 17:38:23 2018

@author: dell 1
"""

#hierarichal clustering
#import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import dataset
dataset=pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,3:5].values

#using dendograms find optimal no of clusters
import scipy.cluster.hierarchy as sch
dendogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('dendogram')
plt.xlabel('customers')
plt.ylabel('euclidean distance')
plt.show()
#fitting the hierarichal cluster to dataset
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(X)
#visualizing the clusters
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,color='red',label='cluster1')    
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,color='blue',label='cluster2') 
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,color='green',label='cluster3') 
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,color='cyan',label='cluster4') 
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,color='magenta',label='cluster5')    
plt.title('clusters of customer')
plt.xlabel('income')
plt.ylabel('spending score')
plt.show()