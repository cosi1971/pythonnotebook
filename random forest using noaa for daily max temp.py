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
data1.shape
#%% 
data1.describe()
data1.duplicated()
#%%
data=data1.dropna()
#%%
data_summary = data.describe()
data_summary
#%%
data['actual'].plot()
data['temp1'].plot()
data['temp2'].plot()
#%% look for Correlation using correlational table and scatter plot matrix 
corr=data.corr()
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
scatter_matrix(data)
plt.show()
#%%
plotlist= (['temp_2', 'temp_1', 'average', 'actual','forecast_noaa', 'forecast_acc', 'forecast_under', 'friend'])
for yaxis in plotlist:
    plt.xlabel('month')
    plt.ylabel(yaxis)
    plt.legend(loc='best')
    plt.plot (data['month'],data[yaxis],label=yaxis)
    plt.show()
#%%
for yaxis in plotlist:
    plt.xlabel('month')
    plt.ylabel(yaxis)
    plt.legend(loc='best')
    plt.scatter (data['month'],data[yaxis],label=yaxis)
    plt.show()

#%%
weekday = (['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])
for day in weekday:
   data['week_'+str(day)]=data['week'].map({str(day):1})

#%% separate the target column from the feature columns, convert to array
import numpy as np
labels=np.array(data['actual'])
data_feat=data.drop(['actual','week','forecast_noaa','forecast_acc','forecast_under'],axis=1)
data_feat=data_feat.fillna(0)
data_feat_row_list=list(data_feat.index)
#%% add the geometric mean of average and temp1 as interaction variable 
data_feat['avg_temp1']=data_feat.average**.5*data_feat.temp_1**.5
data_feat_list = list(data_feat.columns)

#%%
data_feat=np.array(data_feat)
#%% split the feature and target into train and test set
from sklearn.model_selection import train_test_split as md_split
train_feat, test_feat, train_labels, test_labels = md_split(data_feat, labels, test_size = 0.25, random_state = 42)
#%% double check that the sets are split correctly
print ('Training feature shape:',train_feat.shape)
print ('Test feature shape:', test_feat.shape)
print ('Training labels shape:',train_labels.shape)
print ('Test labels shape:', test_labels.shape)

#%% establish a baseline as a way to determine how best we want the trained models to achieve
#use the historical average as the baseline, that is, we want the trained model to perform better than the average)
# to call out the column of avearage in np array, use integer index - the first is the rows, the second is for the columns, like a matrix, but the row and column indices starts with 0, just as the list index
baseline_preds=test_feat[:,data_feat_list.index('average')]
baseline_errors=abs(baseline_preds-test_labels)
abs_avg_error=np.mean(baseline_errors)
chsq_error=(np.mean((baseline_errors)**2))**.5
from scipy.stats import chisquare as chisq
chisq_results = list(chisq(baseline_preds, test_labels))
#%% report whether the baseline is statistically sound - not significant
print ('The goal is to stay within the Average baseline error of', abs_avg_error )
print ('and the mean square error to stay within the chi-square error of', chsq_error)
print ('Chi-square test of difference shows that the test statistic is ',chisq_results[0], 'and the p-value is ',chisq_results[1])
if chisq_results[1] < 0.05:
    print ('and the difference is significant')
else: 
    print ('and the difference is not significant')
#%% Prelim check for most important predictors
from scipy.stats import chisquare as chisq
sig_para=[]
for parameters in data_feat_list:
    baseline_predict=test_feat[:,data_feat_list.index(parameters)]
    baseline_errors_pred=abs(baseline_predict-test_labels)
    abs_avg_errorpred=np.mean(baseline_errors_pred)
    chsq_pred_error=(np.mean((baseline_errors_pred)**2))**.5
    chisq_pred_results = list(chisq(baseline_predict, test_labels))
    if chisq_pred_results[1] >= 0.05:
        sig_para.append(parameters)
'''    print ('Chi-square test of difference shows that the test statistic of ', parameters, 'is ',chisq_pred_results[0])
    print('and the p-value is ',chisq_pred_results[1])
    if chisq_pred_results[1] < 0.05:
        print ('*')
    elif  chisq_pred_results[1] < 0.01:
        print ('**')
'''
print ('The selected significant features are', sig_para)
#%% import random forest model
from sklearn.ensemble import RandomForestRegressor
# instantiate model with 1000 trees, set the RF parameters, train the model by.fit
RF=RandomForestRegressor(n_estimators = 1000, random_state=42)
RF.fit(train_feat, train_labels)
#%% now test the model- Use the forest's predict method on the test data

predictions = RF.predict(test_feat)
# Calculate the absolute errors, the Chi-Square test
errors = abs(predictions - test_labels)
abs_avg_pred_error=np.mean(errors)
chsq_pred_error=(np.mean((errors)**2))**.5
chisq_pred_results = list(chisq(predictions, test_labels))
#%% report whether the baseline is statistically sound - not significant
print ('The Average prediction error is', abs_avg_pred_error )
print ('with the mean square error of', chsq_pred_error)
print ('Chi-square test of difference shows that the test statistic of the prediction is ',chisq_pred_results[0], 'and the p-value is ',chisq_pred_results[1])
if chisq_pred_results[1] < 0.05:
    print ('and the difference is significant')
else: 
    print ('and the difference is not significant')
#%% Visualizing the tree
''' Import tools needed for visualization'''
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = RF.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = r'C:\Users\leeko\Google Drive\Python notebook\forest_tree.dot', feature_names = data_feat_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file(r'C:\Users\leeko\Google Drive\Python notebook\forest_tree.dot')
# Write graph to a png file
graph.write_jpg(r'Google Drive\Python notebook\forest_tree.jpg')
#%%
# Limit depth of tree to 3 levels
RF_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
RF_small.fit(train_feat, train_labels)
# Extract the small tree
tree_small = RF_small.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = r'C:\Users\leeko\Google Drive\Python notebook\small_tree.dot', feature_names = data_feat_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file(r'C:\Users\leeko\Google Drive\Python notebook\small_tree.dot')
graph.write_jpg(r'C:\Users\leeko\Google Drive\Python notebook\small_tree.jpg')
#%%
# Get numerical feature importances
importances = list(RF.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(data_feat_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
#%%so the random forest shows two most important variables
RF_new = RandomForestRegressor(n_estimators= 1000, random_state=42)
# Extract the two most important features
important_indices = [data_feat_list.index('temp_1'), data_feat_list.index('average'),data_feat_list.index('avg_temp1')]
train_important = train_feat[:, important_indices]
test_important = test_feat[:, important_indices]
# Train the random forest
RF_new.fit(train_important, train_labels)
# Make predictions and determine the error
predictions = RF_new.predict(test_important)
errors = abs(predictions - test_labels)
# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = np.mean(100 * (errors / test_labels))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')
#%%
# Get numerical feature importances of significant parameters
importances_new = list(RF_new.feature_importances_)
# List of tuples with variable and importance
feature_importances_new = [(feature, round(importance, 2)) for feature, importance in zip(sig_para, importances_new)]
# Sort the feature importances by most important first
feature_importances_new = sorted(feature_importances_new, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances_new];
#%% Visualization and plotting with predictions by significant predictors


