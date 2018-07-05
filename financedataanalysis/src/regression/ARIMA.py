'''
https://medium.com/@jdwittenauer/a-simple-time-series-analysis-of-the-s-p-500-index-b12ffdb13cd6


stattools.arma_order_select_ic
x13.x13_arima_select_order
arima_model.ARMA
arima_model.ARMAResults

Manual Time Series analysis tsa


Manual Vector Autoregressions tsa.vector_ar

http://devdocs.io/statsmodels/
http://www.statsmodels.org/dev/tsa.html

If stepwise is False, the models will be fit similar to a gridsearch.
auto_arima not to find a model that will converge; if this is the case, it will raise a ValueError.
auto_arima can fit a random search that is much faster than the exhaustive one by enabling random=True.
If your random search returns too many invalid (nan) models, you might try increasing n_fits or making it an exhaustive search.

https://github.com/tgsmith61591/pyramid/blob/master/examples/quick_start_example.ipynb

'''

import sys
import os
import math

import numpy as np
import pandas as pd

import statsmodels.api as sm
import pyramid
from pyramid.arima import auto_arima
from sklearn.metrics import mean_squared_error as mse
import cPickle as pickle

arimamodelrmse = {}
df = pd.read_csv("/home/achivuku/PycharmProjects/financedataanalysis/pricesvolumes.csv")
ArimamodeloutputPath = "/data/achivuku/PycharmProjects/financedataanalysis/arimamodelrmse.pkl"

# print(df)

cols = [1,2,3,4,6,8,10,12,14,16,18,20,21,22,23,24,26,28,30,32,33,34,36,38,40,42]
df.drop(df.columns[cols],axis=1,inplace=True)
df.fillna(0, inplace=True)

# print(len(df.columns))
# print((df.columns))
# print(len(df.index))

allcols = df.columns.tolist()
df[allcols[1:]] = df[allcols[1:]].apply(pd.to_numeric).apply(lambda x: x/x.mean(), axis=0)

# stock = allcols[2] # AAPL_prices
# stock = allcols[3] # ABT_prices
# stock = allcols[4] # AEM_prices
# stock = allcols[5] # AFG_prices
# stock = allcols[6] # APA_prices
# stock = allcols[8] # CAT_prices
# stock = allcols[10] # MCD_prices
# stock = allcols[11] # MSFT_prices
# stock = allcols[12] # ORCL_prices
# stock = allcols[14] # T_prices
# stock = allcols[15] # UTX_prices
stock = allcols[16] # WWD_prices


# for stock in allcols[1:]:
    # stock = allcols[2]

stock_data = df[['Date', stock]]
stock_data = stock_data.set_index('Date')

# stock_data = df[[stock]]

inputdata = stock_data[[stock]].values.flatten()
numrecords = len(inputdata)

numtrainrecords = int(math.ceil(0.7*numrecords))
numtestrecords = int(math.ceil(0.3*numrecords))

rs_fit = auto_arima(inputdata[:numtrainrecords], start_p=1, start_q=1, max_p=10, max_q=10, m=12,
                    start_P=0, seasonal=True, n_jobs=-1, d=1, D=1, trace=True,
                    error_action='ignore',  # don't want to know if an order does not work
                    suppress_warnings=True,  # don't want convergence warnings
                    stepwise=False, random=True, random_state=42,  # we can fit a random search (not exhaustive)
                    n_fits=25)

# print(rs_fit.summary())
predictions = rs_fit.predict(n_periods=numtestrecords)
rmsep = math.sqrt(mse(predictions,inputdata[-numtestrecords:]))
print('Stock: %s, RMSE: %.3f' % (stock,rmsep))


arimamodelrmse['AAPL_prices'] = 0.807
arimamodelrmse['ABT_prices'] = 0.748
arimamodelrmse['AEM_prices'] = 1.643
arimamodelrmse['AFG_prices'] = 0.795
arimamodelrmse['APA_prices'] = 2.795
arimamodelrmse['CAT_prices'] = 1.254
arimamodelrmse['MCD_prices'] = 0.319
arimamodelrmse['MSFT_prices'] = 1.555
arimamodelrmse['ORCL_prices'] = 0.190
arimamodelrmse['T_prices'] = 0.786
arimamodelrmse['UTX_prices'] = 0.209
arimamodelrmse['WWD_prices'] = 0.237

# print(arimamodelrmse)

with open(ArimamodeloutputPath, 'wb') as handle:
    pickle.dump(arimamodelrmse, handle, protocol=pickle.HIGHEST_PROTOCOL)