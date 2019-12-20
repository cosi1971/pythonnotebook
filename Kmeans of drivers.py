# -*- coding: utf-8 -*-
"""
Created on Sat May 25 22:27:31 2019

@author: leeko
"""

#My python script for K means clustering of a 2 dimensional data
import sys
import numpy
import scipy
import pandas as pd
import sklearn.cluster
import matplotlib
from matplotlib import pyplot as plt
url = "https://raw.githubusercontent.com/datascienceinc/learn-data-science/master/Introduction-to-K-means-Clustering/Data/data_1024.csv"
df=pd.read_csv(url,sep = "\t")
#%%
"""
       Driver_ID  Distance_Feature  Speeding_Feature
0     3423311935             71.24              28.0
1     3423313212             52.53              25.0
…
3998  3423313630            176.14               5.0
3999  3423311533            168.03               9.0

[4000 rows x 3 columns]
"""
#%%
x=df.drop('Driver_ID',axis=1)
from sklearn.cluster import KMeans
x
"""
      Distance_Feature  Speeding_Feature
0                71.24              28.0
1                52.53              25.0
...                ...               ...
3998            176.14               5.0
3999            168.03               9.0

[4000 rows x 2 columns]
"""
#%%
X=x.values
kmeansx=KMeans(n_clusters=2).fit(X)
centers=kmeansx.cluster_centers_
centerlabels=kmeansx.labels_
#%%
#cluster labels 
centerlabels

"""array([0, 0, 0, ..., 1, 1, 1])
"""
plt.scatter(X[:,0],X[:,1], c=centerlabels)
plt.show()
#%%
#number of clusters = 4
kmeansx=KMeans(n_clusters=4).fit(X)
centersx=kmeansx.cluster_centers_
centerlabelsx=kmeansx.labels_
plt.scatter(X[:,0],X[:,1], c=centerlabels)
plt.scatter(centers[:,0],centers[:,1], c='black')
plt.show()
#%%
from scipy.spatial.distance import cdist
distortions = []
K = range(1,10)
for k in K:
	kmeany = KMeans(n_clusters=k).fit(X)
	distortions.append(sum(numpy.min(cdist(X, kmeany.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
distortions
#[43.638279319447605, 14.048488898130342, 12.820183191663288, 10.664598030262017, 8.584166997522017, 7.579131349840909, 6.664819785930902, 6.3113917854296036, 6.183746109086731]
#%%# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
#%%
from scipy.spatial.distance import cdist
distortions = []
K = range(1,10)
for k in K:
	kmeany = KMeans(n_clusters=k).fit(X)
	distortions.append(sum(numpy.min(cdist(X, kmeany.cluster_centers_, 'euclidean'), axis=1)) / Y.shape[0])
distortions
[43.638279319447605, 14.048488898130342, 12.820183191663288, 10.664598030262017, 8.584166997522017, 7.579131349840909, 6.664819785930902, 6.3113917854296036, 6.183746109086731]
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

