# -*- coding: utf-8 -*-
"""
Created on Sun May 26 12:36:05 2019

@author: leeko
"""

'''#%%
import sklearn
from sklearn import datasets 
#Load dataset 
wine = datasets.load_wine()
#Import knearest neighbors Classifier model 
from sklearn.neighbors import KNeighborsClassifier 
model = KNeighborsClassifier(n_neighbors=3) 
# Train the model using the training sets 
model.fit(features,label) 
#Predict Output 
predicted= model.predict([[0,2]]) 
# 0:Overcast, 2:Mild print(predicted)
'''
#%%
# Import train_test_split function from sklearn.model_selection 
from sklearn.model_selection import train_test_split 
# Split dataset into training set and test set 
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3) 
# 70% training and 30% test

from sklearn.neighbors import KNeighborsClassifier 
#Create KNN Classifier 
knn = KNeighborsClassifier(n_neighbors=5) 
#Train the model using the training sets 
knn.fit(X_train, y_train) 
#Predict the response for test dataset 
y_pred = knn.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation 
from sklearn import metrics 
# Model Accuracy, how often is the classifier correct? 
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#%%
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import seaborn as sns

kfold=KFold(n_splits=5, random_state=7)
cv_results = []
cv_results.append(['recall', cross_val_score(knn,X_train, y_train,cv=kfold,scoring = 'recall_macro')])
cv_results.append(['precision',cross_val_score(knn,X_train, y_train,cv=kfold,scoring = 'precision_macro')])
cv_results.append(['f1',cross_val_score(knn,X_train, y_train,cv=kfold,scoring = 'f1_macro')])
cv_results.append(['accuracy', cross_val_score(knn,X_train, y_train,cv=kfold,scoring = 'accuracy')])
sns.heatmap (np.asarray(cv_results))
#%%
y_test.size

#%%
cf_mat = metrics.confusion_matrix(y_test,y_pred)/y_test.size
cf_mat
import seaborn as sns
sns.heatmap (cf_mat, annot=True,cmap='viridis')