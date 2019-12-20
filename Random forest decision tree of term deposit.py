# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 14:55:49 2019
#using the banking data, try random forest and see if it works better than logistic regression
#data is balanced using SMOTE
@author: pohch
"""
#%%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
#%%
data= r'Google Drive\Python notebook\bank-additional-full.csv'
df=pd.read_csv(data, sep = ';')
df_backup=df
df['y']=df['y'].map({'no':0,'yes':1})
#%%
df['education']=np.where(df['education'] =='basic.9y', 'Basic', df['education'])
df['education']=np.where(df['education'] =='basic.6y', 'Basic', df['education'])
df['education']=np.where(df['education'] =='basic.4y', 'Basic', df['education'])
df['education'].unique()
df_summary=df.dtypes
df_cont_col=[i for i in df_summary.index if df_summary[i]!='object']
df_final=df[df_cont_col]
#%% create dummy variables from categorical variables and concat to the existing continuous variables
for catg in ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']:
     df_final = pd.concat([df_final,pd.get_dummies(df[catg], prefix=catg)],axis=1) #all in one line one-hot encoding the categorical variables
#%% finalized df
df_final.columns
#%% SMOTE to over-sample the yes-population
# split the predictor set from the target set
X = df_final.loc[:, df_final.columns != 'y']
y = df_final.loc[:, df_final.columns == 'y']
#%% start over-sampling by importing SMOTE (Synthetic Minority Oversampling Technique)
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
os = SMOTE(random_state=0) #os is the over-sampled operator
#train_test_split on predictors X and target Y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns #columns is a list of the predictor labels
os_data_X,os_data_y = os.fit_sample(X_train, y_train) #oversampling operator acts on the predictors and target of training set
os_data_X = pd.DataFrame(data=os_data_X,columns=columns) #convert the numpy array into dataframe objects
os_data_y= pd.DataFrame(data=os_data_y,columns=['y']) #convert the numpy vector into dataframe objects (series)
#%% check the over-sample is balanced
print("length of oversampled data is ",len(os_data_X)) #len is length of the oversample training set 
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))
#%% Recursive feature selection for regresssion
from sklearn.linear_model import LogisticRegression as LR
df_final_vars=df_final.columns.values.tolist()
y=['y']
X=[i for i in df_final_vars if i not in y]
from sklearn.feature_selection import RFE
logreg = LR(solver='liblinear', max_iter=200)
rfe = RFE(logreg, 20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)
#%%
rfe_result= pd.DataFrame(list(zip(X_train.columns.values,rfe.support_,rfe.ranking_)),columns=['predictor', 'yes', 'rank'])
rfe_selected=rfe_result[rfe_result['yes']==1].predictor
#%%
rfe_selected
#%%
rfe_selected=[ele for ele in rfe_selected if ele not in {'marital_unknown', 'default_no', 'default_unknown', 'contact_cellular', 'contact_telephone', 'poutcome_failure', 'poutcome_success', 'poutcome_nonexistent'}]
X=os_data_X[rfe_selected]
y=os_data_y['y']
#%%
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LR(solver='lbfgs', max_iter=200)
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
#%%
from sklearn.metrics import confusion_matrix
confusion_matrix_lr = confusion_matrix(y_test, y_pred_lr)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_lr))
#%% run random forest model
#Running new regression on training data
treeclass = RandomForestClassifier(n_estimators=1000)
treeclass.fit(X_train, y_train)
#Calculating the accuracy of the training model on the testing data
y_pred_rf = treeclass.predict(X_test)
y_pred_rf_prob = treeclass.predict_proba(X_test)
accuracy = treeclass.score(X_test, y_test)
print('The accuracy is: '+ str(accuracy *100) + '%')
#%%
confusion_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print(confusion_matrix_rf)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_rf))
#%%
from sklearn.tree import export_graphviz
import pydot# Pull out one tree from the forest
#%%
tree = treeclass.estimators_[5]# Export the image to a dot file
export_graphviz(tree, out_file = r'Google Drive\Python Notebook\tree.dot', feature_names = rfe_selected, rounded = True, precision = 1)# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file(r'Google Drive\Python Notebook\tree.dot')# Write graph to a png file
graph.write_jpg(r'Google Drive\Python Notebook\tree.jpg')




