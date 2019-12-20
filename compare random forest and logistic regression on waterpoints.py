# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:20:17 2019
using logistic regression and random forest as model selection
@author: leeko
"""
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
#%%
#plots backend
df_train = pd.read_csv(r'Google Drive\Python notebook\Training Set Values.csv')
df_test = pd.read_csv(r'Google Drive\Python notebook\Test Set Values.csv')
train_labels = pd.read_csv(r'Google Drive\Python notebook\Training Set Labels.csv')
df_train.head()
#%% Concatenate the train and test set to get back the full set; to clean up data, 
df_train['training set']=True
df_test['training set']=False
df_full=pd.concat([df_train, df_test])
df_full.shape
df_full.isna().sum()

#df_full.describe
#%% clean up the na entries
train_labels['status_group'].value_counts(normalize=True)
import seaborn as sns
fig, ax = plt.subplots(figsize=(6,8))
sns.countplot(x='status_group', data=train_labels, palette ='hls')
#%% need an 'age' variable since it is very related
# extract the year
df_full['date_recorded'] = pd.to_datetime(df_full['date_recorded'])
df_full['date_recorded'] = df_full['date_recorded'].dt.year
df_full[df_full.construction_year == 0].shape
#df_full[df_full.construction_year >0].construction_year.plot(kind='box')
df_full[df_full.construction_year >0].construction_year.plot(kind='hist', bins=50)
# Replacing the NaN value with the mode - this would turn out to be 
# less effective than I thought
#df_full['construction_year'] = df_full['construction_year'].replace(0, 1986)df_full['age'] = np.abs(df_full['date_recorded'] - df_full['construction_year'])
#%%
df_full[df_full.construction_year == 0].shape
df_full.columns
#%%
df_count=df_full[df_full.construction_year > 0].construction_year.value_counts()
df_count.index
df_count=pd.DataFrame(data=[df_count.index,df_count])
#df_count.plot.bar(x=)