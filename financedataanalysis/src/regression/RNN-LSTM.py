'''
Created on 6 Mar. 2018

2018-03-06 15:42:43.698472: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1032] Ignoring gpu device (device: 1, name: Quadro K2200, pci bus id: 0000:03:00.0) with Cuda multiprocessor count: 5.
The minimum required count is 8. You can adjust this requirement with the env var TF_MIN_GPU_MULTIPROCESSOR_COUNT.

[achivuku@atlas1 financedataanalysis]$ export TF_MIN_GPU_MULTIPROCESSOR_COUNT=5
[achivuku@atlas1 financedataanalysis]$ python RNN-LSTM.py

@author: 99176493
'''
import sys
import os
import math

import numpy as np
import pandas as pd

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, GRU

from sklearn.metrics import mean_squared_error as mse
import cPickle as pickle

import itertools

lstmmodelrmse = {}
LstmmodeloutputPath = "/data/achivuku/PycharmProjects/financedataanalysis/lstmmodelrmse.pkl"


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# fit an LSTM network to univariate training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(2*neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True, return_sequences=True))
	model.add(Dropout(0.5))
	model.add(LSTM(neurons))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
		model.reset_states()

	return model

# make one univariate forecast with an LSTM,
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]


df = pd.read_csv("/home/achivuku/PycharmProjects/financedataanalysis/pricesvolumes.csv")
cols = [1,2,3,4,6,8,10,12,14,16,18,20,21,22,23,24,26,28,30,32,33,34,36,38,40,42]

df.drop(df.columns[cols],axis=1,inplace=True)
df.fillna(0, inplace=True)

allcols = df.columns.tolist()


# for stock in allcols[1:]:
#     stock_data = df[['Date', stock]]
#     stock_data = stock_data.set_index('Date')
#
#     raw_values = stock_data[[stock]].values.flatten()
#     diff_values = difference(raw_values, 1)
#     supervised = timeseries_to_supervised(diff_values, 1)
#     supervised_values = supervised.values
#
#     numrecords = len(supervised_values)
#     numtrainrecords = int(math.ceil(0.7 * numrecords))
#     numtestrecords = int(math.ceil(0.3 * numrecords))
#
#     train, test = supervised_values[:numtrainrecords], supervised_values[-numtestrecords:]
#
#     print('Calling fit_lstm')
#     scaler, train_scaled, test_scaled = scale(train, test)
#     lstm_model = fit_lstm(train_scaled, 1, 15, 25)
#     train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
#     lstm_model.predict(train_reshaped, batch_size=1)
#     predictions = list()
#     for i in range(len(test_scaled)):
#         X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
#         yhat = forecast_lstm(lstm_model, 1, X)
#         yhat = invert_scale(scaler, X, yhat)
#         yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
#         predictions.append(yhat)
#         expected = raw_values[len(train) + i]
#         print('Predicted=%f, Expected=%f' % (yhat, expected))
#     rmsep = math.sqrt(mse(predictions,raw_values[-numtestrecords:]))
#     print('Test RMSE: %.3f' % rmsep)
#     lstmmodelrmse[stock] = rmsep
#
#
# print(lstmmodelrmse)
# with open(LstmmodeloutputPath, 'wb') as handle:
#     pickle.dump(lstmmodelrmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
# {'B_prices': 0.5558748535369104, 'LAKE_prices': 0.49228170904702695, 'AAPL_prices': 1.449782165137462, 'APA_prices': 1.5588648167532124, 'SUN_prices': 4.735098744683062, 'ABT_prices': 0.4618255059989125, 'WWD_prices': 0.8174618558006435, 'AEM_prices': 1.1159182942038577, 'T_prices': 0.3358707177634416, 'UTX_prices': 1.1106567763335713, 'AFG_prices': 0.5807981206050161, 'MSFT_prices': 0.597965092461479, 'ORCL_prices': 0.521133463568524, 'MCD_prices': 0.981582075715957, 'IXIC_prices': 40.26938733818711, 'CAT_prices': 1.4748489482894944}


# X, y = train_scaled[:, 0:-1], train_scaled[:, -1]
# X = X.reshape(X.shape[0], 1, X.shape[1])
# print(X)
# sys.exit()

# from pandas import read_csv
# def parser(x):
# 	return datetime.strptime('190'+x, '%Y-%m')
# series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# X = series.values
# train, test = X[0:-12], X[-12:]
# print(train)
# print(train.shape)
# print(test)
# print(test.shape)




# supervised = timeseries_to_supervised(X, 1)
# print(supervised.head())
# sys.exit()

# differenced = difference(X, 1)
# print(differenced.head())
# sys.exit()

# inverted = list()
# for i in range(len(differenced)):
#     value = inverse_difference(X, differenced[i], len(X)-i)
#     inverted.append(value)
# inverted = Series(inverted)
# print(inverted.head())


# raw_values = raw_values.reshape(len(raw_values), 1)
# scaler = MinMaxScaler(feature_range=(-1, 1))
# scaler = scaler.fit(raw_values)
# scaled_X = scaler.transform(raw_values)

# scaled_series = Series(scaled_X[:, 0])
# print(scaled_series.head())
# inverted_X = scaler.inverse_transform(scaled_X)
# print(scaled_X)
# print(inverted_X)

