# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:41:47 2019

@author: leeko
This program uses the bayesian approach to predict whether customers will go for term deposit based on posterior data
that is, prob (certain characteristics | enrolled in deposit program)
the second idea is that with all the data given, and the situation, we do know the posterior frequency
and hence the posterior probability prob(enrolled in program | certain traits)
but we want to trace the probability distrbution, prior of program enrolment from posterior of program enrolment
the third idea is that such a distribution is difficult to retrieve from catergorical outcomes, enrolled or not enrolled
therefore the approach is to use odd ratio, logit, regression of it from the posterior data points
the algebra calculations of such intergrations is not easy. so by using numerical methods is more feasiable
hence the use of PYMC3's monte carlo sampling method as the numerical appraoch to reconstruct the prior distribution of the logit of enrolment
from the poster logit of enrolment data.
when you have a lot of data points, they are useful enough to build the distribution curve
"""
#%%
''' the following starts with a series of prior data exploration by visual comparison of various continuous variables between the sign up and the non-sign up 
the goals is to identify promising predictors/features that is likely to predict the sign-up rate
'''
import pandas as pd
import scipy as sp
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az
import matplotlib.lines as mlines
import warnings
from collections import OrderedDict
import theano
import itertools
from IPython.core.pylabtools import figsize
pd.set_option('display.max_columns', 30)
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
df=pd.read_csv(r'Google Drive\Python notebook\bank-full.csv', sep=';' )
df.columns.values[16]='deposit' #change the column label 'y' to 'deposit'
df.columns.values[16]
#%%
#explore the statistical difference between those who sign up and those not
df_small=df
df_summary=df.dtypes
df_summary
df_shape=df.shape
df_shape
df.describe()
#%% the EDA has been done in another similar program "Bayesian logistic regression of term deposit"
'''print ('Continuous variables frequencies')
for catg in df_summary.index:
    if df_summary[catg] == 'int64':
        plt.hist(df[catg], density= True, bins=30)
        plt.ylabel(catg)
    plt.show()
for catg in df_summary.index:
    if df_summary[catg] == 'float64':
        plt.hist(df[catg], density= True, bins=30)
        plt.ylabel(catg)
    plt.show()
    '''
#%%
print ('Categorical variables frequencies')
for catg in df_summary.index:
    if df_summary[catg] == 'object':
        table=df[catg].value_counts()
        print(table)
        print()
#        plt.bar(table.index, table, label=catg)
#        plt.ylabel(catg)
#        df[catg].describe()
#    plt.show()
#%%
no balance column in the bank-additional-full csv
df_select=pd.DataFrame(df['deposit'])
df_select['balance']=df['balance']
df_select[df_select.deposit== 'yes'].balance.plot(kind='hist', bins=30)
df_select[df_select.deposit== 'no'].balance.plot(kind='hist',bins=30)
'''
#%%
# use side-by-side histogram plot of continuous variable to explore possible significant differences
for catg in df_summary.index:
    if df_summary[catg] == 'int64':
        df_select=pd.DataFrame(df['deposit'])
        df_select[catg]=df[catg]
        ax_yes=plt.subplot(211)
        df_select_yes = df_select[df_select.deposit== 'yes']
        ax_yes.hist(df_select_yes[catg], bins=100, color = 'blue')
        ax_no=plt.subplot(212)
        df_select_no = df_select[df_select.deposit== 'no']
        ax_no.hist(df_select_no[catg], bins=100, color='green')
        ax_yes.set_title('Histogram with %s' % catg, size = 20)
        ax_no.set_xlabel(catg, size = 16)
        ax_no.set_ylabel('Frequency', size= 16)
        ax_yes.set_ylabel('Frequency', size= 16)
        plt.show()        
#    df_select[df_select.deposit== 'yes'].plot(kind='hist', bins=180)
#    df_select[df_select.deposit== 'no'].plot(kind='hist',bins=180)
#%%
# use superimposed distribution plot of continuous variable to explore possible significant differencesfor catg in df_summary.index:
# run t-test or z-test compare between yes and no      
import statsmodels.stats.weightstats as stmodel
for catg in df_summary.index:
    if df_summary[catg] == 'int64':
        df_select=pd.DataFrame(df['deposit'])
        df_select[catg]=df[catg]
        plt.figure(figsize=(12,6))
        df_select_yes = df_select[df_select.deposit== 'yes']
        sns.distplot(df_select_yes[catg], hist=False, norm_hist=True, color='blue', label='sign up deposit')
        df_select_no = df_select[df_select.deposit== 'no']
        sns.distplot(df_select_no[catg], hist=False, norm_hist=True, color ='red', label='no sign up')
        plt.title("comparison of "+catg, size=20)
        plt.show()
#        tstat, p_value=stats.ttest_ind(df_select_yes[catg],df_select_no[catg])
        tstat, p_value=stmodel.ztest(df_select_yes[catg],df_select_no[catg])
        print("t-statistic is", tstat, "and the p-value is",p_value)
        if p_value < 0.05:
            print("Significant predictor")
'''    
#%% begin the bayesian logistic regression
# first convert the education, job, marital status, default, housing,  replace, and others to ordinal values
df_new=df
df_new['job']=df['job'].map({'management':12, 'admin.':11,'entrepreneur':10,'technician':9,'services':8,'self-employed':7,'blue-collar':6,'retired':5, 'housemaid':4,'unemployed':3,'unknown':2,'student':1})
df_new['marital']=df['marital'].map({'married':3, 'single':2, 'divorced':1})
df_new['education'] = df['education'].map({'unknown': 1, 'primary': 2, 'secondary':3, 'tertiary':4})
df_new['default']=df['default'].map({'no':0, 'yes':1})
df_new['housing']=df['housing'].map({'no':0, 'yes':1})
df_new['loan']=df['loan'].map({'no':0, 'yes':1})
df_new['contact']=df['contact'].map({'unknown': 0, 'cellular':1, 'telephone':2})
df_new['poutcome']=df['poutcome'].map({'failure':0, 'success':1,'unknown':2, 'other':3})
df_new['month']=df['month'].map({'jan':1, 'feb':2,'aug': 8, 'nov': 11, 'jun': 6, 'apr': 4, 'jul': 7,'may': 5, 'oct': 10, 'mar': 3, 'sep': 9, 'dec': 12})
#df_new['day_of_week']=df['day_of_week'].map({'thu': 4, 'fri': 5, 'tue': 2, 'mon': 1, 'wed': 3})
df_new['deposit']=df['deposit'].map({'no':0, 'yes':1})
#%%
df_new_chk=df_new.isna().sum()# check that every text input is converted to a ordinal value
for label in df_new_chk.index:
    if df_new_chk[label] != 0:
        print ('error in converting ', label) 
#%%
df_new.columns
sns.countplot(x='deposit', data=df_new, palette='hls')
plt.show()
plt.savefig(r'Google Drive\Python notebook\bank-count-plot')
#%%
count_no_sub = len(df_new[df_new['deposit']==0])
count_sub = len(df_new[df_new['deposit']==1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no subscription is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of subscription", pct_of_sub*100)
#%%
for catg in df_summary.index:
    if df_summary[catg] == 'object':
        print(df.groupby(catg).mean())
