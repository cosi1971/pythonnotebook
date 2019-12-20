# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 13:40:20 2019

@author: leeko
"""

from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()                        # Load iris dataset

X = iris.data[:, [2, 3]]                           # Assign matrix X
y = iris.target                                    # Assign vector y
#%%
from sklearn.tree import DecisionTreeClassifier    # Import decision tree classifier model

tree = DecisionTreeClassifier(criterion='entropy', # Initialize and fit classifier
    max_depth=4, random_state=1)
tree.fit(X, y)
#%%
from pydotplus.graphviz import graph_from_dot_data
from sklearn.tree import export_graphviz

dot_data = export_graphviz(                           # Create dot data
    tree, filled=True, rounded=True,
    class_names=['Setosa', 'Versicolor','Virginica'],
    feature_names=['petal length', 'petal width'],
    out_file=None)

graph = graph_from_dot_data(dot_data)                 # Create graph from dot data
graph.write_jpg(r'C:\Users\leeko\Google Drive\Python notebook\tree.jpg')                           # Write graph to PNG image
#%%
"""
run regression tree 
"""
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

boston = datasets.load_boston()            # Load Boston Dataset
df = pd.DataFrame(boston.data[:, 12])      # Create DataFrame using only the LSAT feature
df.columns = ['LSTAT']
df['MEDV'] = boston.target                 # Create new column with the target MEDV
df.head()
#%%
from sklearn.tree import DecisionTreeRegressor    # Import decision tree regression model

X = df[['LSTAT']].values                          # Assign matrix X
y = df['MEDV'].values                             # Assign vector y
#%%
sort_idx = X.flatten().argsort()                  # Sort X and y by ascending values of X
X = X[sort_idx]
y = y[sort_idx]

tree = DecisionTreeRegressor(criterion='mse',     # Initialize and fit regressor
                             max_depth=3)         
tree.fit(X, y)

#%%
plt.figure(figsize=(7, 6))
plt.scatter(X, y, c='steelblue',                  # Plot actual target against features
            edgecolor='white', s=70)
plt.plot(X, tree.predict(X),                      # Plot predicted target against features
         color='black', lw=2)
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000s [MEDV]')
plt.show()


#%%
from pydotplus.graphviz import graph_from_dot_data
from sklearn.tree import export_graphviz

dot_data = export_graphviz(                           # Create dot data
    tree, filled=True, rounded=True,
    class_names=['low', 'middle','high'],
    feature_names=['pop'],
    out_file=None)

graph = graph_from_dot_data(dot_data)                 # Create graph from dot data
graph.write_jpg(r'C:\Users\leeko\Google Drive\Python notebook\Rtree.jpg')                           # Write graph to PNG image
#%%
ypred=tree.predict(X)
XYcoord=pd.DataFrame(X, columns=['pop'])
XYcoord['Median']=pd.Series(ypred)
plt.plot(XYcoord['pop'],XYcoord['Median'], color='black',marker='o')
#%%
import numpy as np
xcoord=X.flatten()
ycoord=np.array(tree.predict(X))
xycoord= pd.DataFrame({'pop':xcoord,'Median':ycoord})
plt.scatter(X, y, c='steelblue', edgecolor='red', s=20, marker='^')
plt.plot(xycoord['pop'],xycoord['Median'], color='black')
