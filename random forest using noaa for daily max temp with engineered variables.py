# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 20:11:42 2019

@author: leeko
This is a test using random forest for climate data

"""
#%%
import pandas as pd
data1 = pd.read_csv(r'Google Drive\Python notebook\temps.csv')
data1.head()
#%%
data=data1.dropna()
#%%
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
weekday = (['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])
for day in weekday:
   data['week_'+str(day)]=data['week'].map({str(day):1})

#%% separate the target column from the feature columns, convert to array
import numpy as np
labels=np.array(data['actual'])
data_feat0=data.drop(['actual','week','forecast_noaa','forecast_acc','forecast_under'],axis=1)
data_feat0=data_feat0.fillna(0)
data_feat_row_list=list(data_feat0.index)
#%% add the geometric mean of average and temp1 as interaction variable 
#engineered variable here
from sklearn.model_selection import train_test_split as md_split
sig_para=['temp_1','average','avg_temp1']
from sklearn.ensemble import RandomForestRegressor
#%%
for n in range(0,20):
    data_feat0['avg_temp1']=n/10*data_feat0.average**.5*data_feat0.temp_1**.5

    data_feat_list = list(data_feat0.columns)
#conversion to array
    data_feat=np.array(data_feat0)
#split the feature and target into train and test set
    train_feat, test_feat, train_labels, test_labels = md_split(data_feat, labels, test_size = 0.25, random_state = 42)
#Prelim check for most important predictors
#from scipy.stats import chisquare as chisq
    '''for parameters in data_feat_list:
    baseline_predict=test_feat[:,data_feat_list.index(parameters)]
    baseline_errors_pred=abs(baseline_predict-test_labels)
    abs_avg_errorpred=np.mean(baseline_errors_pred)
    chsq_pred_error=(np.mean((baseline_errors_pred)**2))**.5
    chisq_pred_results = list(chisq(baseline_predict, test_labels))
    if chisq_pred_results[1] >= 0.05:
        sig_para.append(parameters)
    print ('Chi-square test of difference shows that the test statistic of ', parameters, 'is ',chisq_pred_results[0])
    print('and the p-value is ',chisq_pred_results[1])
    if chisq_pred_results[1] < 0.05:
        print ('*')
    elif  chisq_pred_results[1] < 0.01:
        print ('**')
print ('The selected significant features are', sig_para)'''
#import random forest model
# instantiate model with 1000 trees, set the RF parameters, train the model by.fit
    RF=RandomForestRegressor(n_estimators = 1000, random_state=42)
    RF.fit(train_feat, train_labels)
#so the random forest shows two most important variables
    RF_new = RandomForestRegressor(n_estimators= 1000, random_state=42)
# Extract the two most important features
    important_indices = [data_feat_list.index(sig_para[0]), data_feat_list.index(sig_para[1]),data_feat_list.index(sig_para[2])]
    train_important = train_feat[:, important_indices]
    test_important = test_feat[:, important_indices]
# Train the random forest
    RF_new.fit(train_important, train_labels)
# Make predictions and determine the error
    predictions = RF_new.predict(test_important)
    errors = abs(predictions - test_labels)
# Display the performance metrics
    print('For the n value of', n)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    mape = np.mean(100 * (errors / test_labels))
    accuracy = 100 - mape
    print('Accuracy:', round(accuracy, 2), '%.')
#
# Get numerical feature importances of significant parameters
    importances_new = list(RF_new.feature_importances_)
# List of tuples with variable and importance
    feature_importances_new = [(feature, round(importance, 2)) for feature, importance in zip(sig_para, importances_new)]
# Sort the feature importances by most important first
    feature_importances_new = sorted(feature_importances_new, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
    print('For the value of ',n) 
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances_new];
#%% Visualization and plotting with predictions by significant predictors


