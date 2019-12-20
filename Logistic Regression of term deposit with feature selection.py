# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:57:59 2019
This is a logistic regression of customer bank data to predict whether who will sign up for term deposit

@author: pohch
"""
#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
plt.rc("font", size =14)
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
sns.set(style = "white")
sns.set(style="whitegrid", color_codes=True)
#%%
df=pd.read_csv(r'Google Drive\Python notebook\bank-additional-full.csv', sep=';' )
df_backup=df
df['y']=df['y'].map({'no':0,'yes':1})
#%%
df_summary=df.dtypes
df_summary
#%%
df['education']=np.where(df['education'] =='basic.9y', 'Basic', df['education'])
df['education']=np.where(df['education'] =='basic.6y', 'Basic', df['education'])
df['education']=np.where(df['education'] =='basic.4y', 'Basic', df['education'])
df['education'].unique()
#%%
df.groupby('job').mean()
#%%
pd.crosstab(df.job,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')
plt.savefig(r'Google Drive\Python Notebook\purchase_fre_job')
table=pd.crosstab(df.job,df.y)
print(table)
#%%
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Education vs Purchase')
plt.xlabel('Education')
plt.ylabel('Proportion of Customers')
plt.savefig('Google Drive\Python Notebook\edu_vs_pur_stack')
#%%
df_summary
#%%
df_cont_col=[i for i in df_summary.index if df_summary[i]!='object']
#%%
df_final=df[df_cont_col]
#%% create dummy variables from categorical variables and concat to the existing continuous variables
for catg in ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']:
#    if df_summary[catg] == 'object':
     df_final = pd.concat([df_final,pd.get_dummies(df[catg], prefix=catg)],axis=1) #all in one line one-hot encoding the categorical variables
#    elif df_summary[catg] != 'object':
#        df = pd.concat(df,df[catg])
#%% finalized df
df_final.columns
#%% SMOTE to over-sample the yes-population
# split the predictor set from the target set
X = df_final.loc[:, df_final.columns != 'y']
y = df_final.loc[:, df_final.columns == 'y']
#start over-sampling by importing SMOTE (Synthetic Minority Oversampling Technique)
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0) #os is the over-sampled operator
#train_test_split on predictors X and target Y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns #columns is a list of the predictor labels
#%%
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
df_final_vars=df_final.columns.values.tolist()
y=['y']
X=[i for i in df_final_vars if i not in y]
from sklearn.feature_selection import RFE
logreg = LR()
rfe = RFE(logreg, 20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)
#%%
rfe_result= pd.DataFrame(list(zip(X_train.columns.values,rfe.support_,rfe.ranking_)),columns=['predictor', 'yes', 'rank'])
rfe_selected=rfe_result[rfe_result['yes']==1].predictor
#%%
X=os_data_X[rfe_selected]
y=os_data_y['y']
#%%
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
result.summary2()
#%%
rfe_selected=[ele for ele in rfe_selected if ele not in {'marital_unknown', 'default_no', 'default_unknown', 'contact_cellular', 'contact_telephone', 'poutcome_failure', 'poutcome_success', 'poutcome_nonexistent'}]
X=os_data_X[rfe_selected]
y=os_data_y['y']
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())
#%%
from sklearn.linear_model import LogisticRegression as LR 
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LR()
logreg.fit(X_train, y_train)
#%%
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
#%%
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
#%%
'''
The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier to not label a sample as positive if it is negative.
The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall, where an F-beta score reaches its best value at 1 and worst score at 0.
The F-beta score weights the recall more than the precision by a factor of beta. beta = 1.0 means recall and precision are equally important.
The support is the number of occurrences of each class in y_test.
'''
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
#%%
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

