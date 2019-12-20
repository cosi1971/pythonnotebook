#!/usr/bin/env python
# coding: utf-8

# In[1]:


#time series analysis 
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,8

importfile = r'C:\Users\leeko\Google Drive\Python notebook\personalISQ.xlsx'
##If from local drive, then use r'C:\....\file_name' since the \ special characters needs to be read like text.
#expense1.xlsx

##If data is csv in the local file, then use this csv import 
# rawdata=pd.read_csv(importfile)

##if data is excel in the local folder
rawdata=pd.read_excel(importfile)

#check data 
rawdata.columns


# In[4]:


#extract the needed columns into the timecol and seriescol and make it a dataframe object
timecol, seriescol= 'Term','Mean'

rawts=rawdata[[timecol,seriescol]].copy()
#convert the dataframe object into a timeseries

from datetime import datetime

rawts[timecol]=pd.to_datetime(rawts[timecol],infer_datetime_format=True)
indexedts=rawts.set_index([timecol])
indexedts.describe
#check any missing data
indexedts.isna()


# In[5]:


indexedts


# In[6]:


# extract the column labels and use as graph labels
vlabel=indexedts.columns.values[0]
hlabel=timecol
plt.xlabel(timecol)
plt.ylabel(vlabel)
plt.plot(indexedts)


# In[7]:


# Set window period according to days, week, months, quarters, years for averaging the rolling mean
winperiod=12
#determine rolling stats using quarter period
rolmean=indexedts.rolling(window=winperiod).mean()
rolstd=indexedts.rolling(window=winperiod).std()
orig = plt.plot(indexedts,color='blue',label = 'balance')
mean = plt.plot(rolmean, color ='green', label='rolling mean')
std = plt.plot(rolstd, color='black',label='rolling std')
plt.legend(loc='right')
plt.title('Rolling Mean and Standard Deviation')
plt.show(block=False)


# In[8]:


# create a stationary test of the original time series

from statsmodels.tsa.stattools import adfuller

#pass the timeseries to be tested, period is the window number and attrib is the name of the series
def test_stationary (timeseries, period):

    #determine rolling stats using the period number
    rolmean=timeseries.rolling(window=period).mean()
    rolstd=timeseries.rolling(window=period).std()

    #set the attrib to the string of the series label
    attrib=timeseries.columns.values[0]

    #plot rollstats
    orig = plt.plot(timeseries,color='blue',label = 'original')
    mean = plt.plot(rolmean, color ='green', label='rolling mean')
    std = plt.plot(rolstd, color='black',label='rolling std')
    plt.legend(loc='right')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)

    #check with dicker fuller test
    print ('Results of Dicker Fuller Test')
    dftest = adfuller(timeseries[attrib],autolag='AIC')
    dfoutput=pd.Series(dftest[0:4],index=['Test-stat','p-value','#lags used','Number of observation used'])

    #set the p-value for stationary
    alpha = 0.05

    #print test outcome
    for key, value in dftest[4].items():
        dfoutput['Critical Value(%s)'%key]=value
    if dfoutput.loc['p-value'] < alpha:
        print ('the series is stationary\n')
    else:
        print ('the series is not stationary\n')
    print (dfoutput) 


# In[9]:


# create a stationary test of the residual of the original time series

from statsmodels.tsa.stattools import adfuller

#pass the timeseries to be converted to residual and then tested, period is the window number)
def test_resstationary (timeseries, period):

    #determine rolling stats using the period number
    rolmean=timeseries.rolling(window=period).mean()
    rolstd=timeseries.rolling(window=period).std()

    #create the residual
    rests=timeseries- rolmean
    rests.dropna(inplace=True)
      
    #plot rollstats
    orig = plt.plot(timeseries,color='blue',label = 'original')
    mean = plt.plot(rolmean, color ='green', label='rolling mean')
    std = plt.plot(rolstd, color='black',label='rolling std')
    resd= plt.plot(rests, color = 'red',label='residual')
    plt.legend(loc='right')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)

    #set the attrib to the string of the series label
    attrib=rests.columns.values[0]

    #check with dicker fuller test
    print ('Results of Dicker Fuller Test')
    dftest = adfuller(rests[attrib],autolag='AIC')
    dfoutput=pd.Series(dftest[0:4],index=['Test-stat','p-value','#lags used','Number of observation used'])

    #set the p-value for stationary
    alpha = 0.05

    #print test outcome
    for key, value in dftest[4].items():
        dfoutput['Critical Value(%s)'%key]=value
    if dfoutput.loc['p-value'] < alpha:
        print ('the series is stationary\n')
    else:
        print ('the series is not stationary\n')
    print (dfoutput) 


# In[15]:


test_stationary(indexedts,winperiod)
#test_resstationary(indexedts,winperiod)


# In[26]:


indexedts


# In[17]:


## to do make all values positive before transformation log 
# to remove the negative balance by translation of minimum and centering of mean

if indexedts.min()[0] <0:
    indexedtspos=indexedts-indexedts.min()+indexedts.mean()
    indexedtspos.min()
else:
    indexedtspos=indexedts

#transform the positive series to logscale
# indexedtslog=np.log(indexedtspos)
plt.plot(indexedts)


# In[13]:


test_stationary(indexedts,winperiod)


# In[18]:


MA=indexedts.rolling(window=winperiod).mean()
Mstd=indexedts.rolling(window=winperiod).std()
indexedtsminusMA=indexedts - MA
indexedtsminusMA.dropna(inplace=True)
test_stationary(indexedtsminusMA,winperiod)


# In[20]:


expdecwAvg=indexedts.ewm(halflife=winperiod,min_periods=0,adjust=True).mean()
plt.plot(indexedts, label=seriescol)
plt.plot(expdecwAvg,color='red',label='weighted Avg')


# In[22]:


indexedtsminusEMA=indexedts-expdecwAvg
test_stationary(indexedtsminusEMA,winperiod)


# In[27]:


#I in ARIMA is the number of times the timeseries is differentiated by shifting
indexedtsdiffshift=indexedts-indexedts.shift()
indexedtsdiffshift.dropna(inplace=True)
plt.plot(indexedtsdiffshift)
test_stationary(indexedtsdiffshift,winperiod)


# In[28]:


# IF original series is tested true for stationary, so need not differentiate.
#url link for ARIMA and steps - https://people.duke.edu/~rnau/411arim2.htm

#import the timeseries library for ARIMA, seasonal_decompose returns three stats, the trend, seasonal and residual
from statsmodels.tsa.seasonal import seasonal_decompose

timeseries=indexedts

decomposition = seasonal_decompose(timeseries)
trend=decomposition.trend
seasonal=decomposition.seasonal
residual=decomposition.resid

#subplot(nrows, ncols, index, **kwargs)
#subplot(pos, **kwargs)
#subplot(ax)
#*args Either a 3-digit integer or three separate integers describing the position of the subplot. If the three integers are nrows, ncols, and index in order, the subplot will take the index position on a grid with nrows rows and ncols columns. index starts at 1 in the upper left corner and increases to the right.
#pos is a three digit integer, where the first digit is the number of rows, the second the number of columns, and the third the index of the subplot. i.e. fig.add_subplot(235) is the same as fig.add_subplot(2, 3, 5). Note that all integers must be less than 10 for this form to work.

plt.subplot(411)
plt.plot(timeseries, label='Original')
plt.legend(loc='right')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='right')
plt.subplot(413)
plt.plot(seasonal, label='Seasonal')
plt.legend(loc='right')
plt.subplot(414)
plt.plot(residual, label='Residual')
plt.legend(loc='right')
plt.tight_layout()


# In[31]:


decomposedts=residual
decomposedts.dropna(inplace=True)
test_stationary (decomposedts,winperiod)


# In[32]:


#ACF and PACF plots, ACF for q of MA, PACF for p of AR
from statsmodels.tsa.stattools import acf, pacf

timeseries=indexedtsdiffshift

lag_acf=acf(timeseries,nlags=20)
lag_pacf=pacf(timeseries,nlags=20,method='ols')

#plot ACF
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle="--",color='gray')
plt.axhline(y=-1.96/np.sqrt(len(timeseries)),linestyle="--",color='gray')
plt.axhline(y=1.96/np.sqrt(len(timeseries)),linestyle="--",color='gray')
plt.title('Autocorrelation Function')

#plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle="--",color='gray')
plt.axhline(y=-1.96/np.sqrt(len(timeseries)),linestyle="--",color='gray')
plt.axhline(y=1.96/np.sqrt(len(timeseries)),linestyle="--",color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


# In[33]:


from statsmodels.tsa.arima_model import ARIMA
timeseries=indexedts

#AR Model
model =ARIMA(timeseries,order = (2,1,2), freq= 'MS')
ARIMAresult= model.fit(disp=-1)
plt.plot(indexedtsdiffshift)
plt.plot(ARIMAresult.fittedvalues, color='red')
plt.title ('RSS: %.4f'%sum((ARIMAresult.fittedvalues-indexedtsdiffshift[seriescol])**2))
print ('plotting ARIMA Model')


# In[ ]:


#Fitting of timeseries model
pred_ARIMA_diff = pd.Series(ARIMAresult.fittedvalues, copy =True)
pred_ARIMA_diff.head()


# In[ ]:


#convert to cumulative sum
pred_ARIMA_diff_cumsum= pred_ARIMA_diff.cumsum()
pred_ARIMA_diff_cumsum.head()


# In[ ]:


pred_ARIMA_log=pd.Series(indexedtslog[seriescol].iloc[0], index=indexedtslog.index)
pred_ARIMA_log=pred_ARIMA_log.add(pred_ARIMA_diff_cumsum, fill_value=0)
pred_ARIMA_log.index[143]


# In[ ]:


# Transform the log series back to the orginal scale, reversing any translation
pred_ARIMA=np.exp(pred_ARIMA_log) #-indexedts[seriescol].mean()+indexedts[seriescol].min()
pred_ARIMA.head()
plt.plot(indexedtspos)
plt.plot(pred_ARIMA)


# In[ ]:


ARIMAresult.plot_predict(1, 264)
x=ARIMAresult.forecast(steps=120)


# In[ ]:


y=np.exp(x[0])
y
# how to create an table
from datetime import timedelta

nowmonth=pred_ARIMA_log.index[143]
forecastperiod=pd.date_range(nowmonth, periods=121, freq='MS')
forecastperiod=forecastperiod[1:].copy()
forecastperiod
predict_table=pd.DataFrame(y,forecastperiod,columns=[seriescol] )
predict_table.index.name=timecol
predict_table

# plt.plot(predict_table)


# In[ ]:





# In[ ]:




