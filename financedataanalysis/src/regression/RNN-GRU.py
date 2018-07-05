'''
Created on 6 Mar. 2018

2018-03-06 15:42:43.698472: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1032] Ignoring gpu device (device: 1, name: Quadro K2200, pci bus id: 0000:03:00.0) with Cuda multiprocessor count: 5.
The minimum required count is 8. You can adjust this requirement with the env var TF_MIN_GPU_MULTIPROCESSOR_COUNT.

[achivuku@atlas1 financedataanalysis]$ export TF_MIN_GPU_MULTIPROCESSOR_COUNT=5
[achivuku@atlas1 financedataanalysis]$ python RNN-GRU.py

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
import networkx as nx
import itertools
import collections

# frame a sequence as a supervised learning problem
# def timeseries_to_supervised(data, lag=1):
#     df = DataFrame(data)
#     columns = [df.shift(i) for i in range(1, lag+1)]
#     columns.append(df)
#     df = concat(columns, axis=1)
#     df.fillna(0, inplace=True)
#     return df

# def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
# 	n_vars = 1 if type(data) is list else data.shape[1]
# 	df = DataFrame(data)
# 	cols, names = list(), list()
# 	# input sequence (t-n, ... t-1)
# 	for i in range(n_in, 0, -1):
# 		cols.append(df.shift(i))
# 		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
# 	# forecast sequence (t, t+1, ... t+n)
# 	for i in range(0, n_out):
# 		cols.append(df.shift(-i))
# 		if i == 0:
# 			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
# 		else:
# 			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
# 	# put it all together
# 	agg = concat(cols, axis=1)
# 	agg.columns = names
# 	# drop rows with NaN values
# 	if dropnan:
# 		agg.dropna(inplace=True)
# 	return agg



# def difference(dataset, interval=1):
#     diff = list()
#     for i in range(interval, len(dataset)):
#         value = dataset[i] - dataset[i - interval]
#         diff.append(value)
#     return Series(diff)

# def inverse_difference(history, yhat, interval=1):
#     return yhat + history[-interval]
#
# def scale(train, test):
# 	# fit scaler
# 	scaler = MinMaxScaler(feature_range=(-1, 1))
# 	scaler = scaler.fit(train)
# 	# transform train
# 	train = train.reshape(train.shape[0], train.shape[1])
# 	train_scaled = scaler.transform(train)
# 	# transform test
# 	test = test.reshape(test.shape[0], test.shape[1])
# 	test_scaled = scaler.transform(test)
# 	return scaler, train_scaled, test_scaled
#
# def invert_scale(scaler, X, value):
# 	new_row = [x for x in X] + [value]
# 	array = np.array(new_row)
# 	array = array.reshape(1, len(array))
# 	inverted = scaler.inverse_transform(array)
# 	return inverted[0, -1]

# # fit an GRU network to multistep training data
# def fit_gru(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
# 	# reshape training into [samples, timesteps, features]
# 	X, y = train[:, 0:n_lag], train[:, n_lag:]
# 	X = X.reshape(X.shape[0], 1, X.shape[1])
# 	# design network
# 	model = Sequential()
# 	# model.add(LSTM(2*n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True, return_sequences=True))
# 	model.add(GRU(2*n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True, return_sequences=True))
# 	model.add(Dropout(0.5))
# 	# model.add(LSTM(n_neurons))
# 	model.add(GRU(n_neurons, stateful=True, return_sequences=True))
# 	model.add(Dropout(0.5))
# 	model.add(GRU(n_neurons, stateful=True, return_sequences=True))
# 	model.add(Dropout(0.2))
# 	model.add(GRU(n_neurons))
# 	model.add(Dropout(0.5))
# 	model.add(Dense(y.shape[1]))
# 	model.compile(loss='mean_squared_error', optimizer='adam')
# 	# fit network
# 	for i in range(nb_epoch):
# 		model.fit(X, y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
# 		model.reset_states()
# 	return model

# # make one univariate forecast with an LSTM,
# def forecast_lstm(model, batch_size, X):
# 	X = X.reshape(1, 1, len(X))
# 	yhat = model.predict(X, batch_size=batch_size)
# 	return yhat[0,0]

# make one multistep forecast with an LSTM,
# def forecast_gru(model, X, n_batch):
# 	# reshape input pattern to [samples, timesteps, features]
# 	X = X.reshape(1, 1, len(X))
# 	# make forecast
# 	forecast = model.predict(X, batch_size=n_batch)
# 	# convert to array
# 	return [x for x in forecast[0, :]]

# evaluate the persistence model
# def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
# 	forecasts = list()
# 	for i in range(len(test)):
# 		X, y = test[i, 0:n_lag], test[i, n_lag:]
# 		# make forecast
# 		forecast = forecast_gru(model, X, n_batch)
# 		# store the forecast
# 		forecasts.append(forecast)
# 	return forecasts

# def inverse_difference(last_ob, forecast):
# 	# invert first forecast
# 	inverted = list()
# 	inverted.append(forecast[0] + last_ob)
# 	# propagate difference forecast using inverted first value
# 	for i in range(1, len(forecast)):
# 		inverted.append(forecast[i] + inverted[i-1])
# 	return inverted
#
# def inverse_transform(series, forecasts, scaler, n_test):
# 	inverted = list()
# 	for i in range(len(forecasts)):
# 		# create array from forecast
# 		forecast = np.array(forecasts[i])
# 		forecast = forecast.reshape(1, len(forecast))
# 		# invert scaling
# 		inv_scale = scaler.inverse_transform(forecast)
# 		inv_scale = inv_scale[0, :]
# 		# invert differencing
# 		index = len(series) - n_test + i - 1
# 		last_ob = series.values[index]
# 		inv_diff = inverse_difference(last_ob, inv_scale)
# 		# store
# 		inverted.append(inv_diff)
# 	return inverted
#
#
# def evaluate_forecasts(test, forecasts, n_lag, n_seq):
# 	for i in range(n_seq):
# 		actual = [row[i] for row in test]
# 		predicted = [forecast[i] for forecast in forecasts]
# 		rmsep = math.sqrt(mean_squared_error(actual, predicted))
# 		print('t+%d RMSE: %f' % ((i+1), rmsep))
# 		return rmsep



df = pd.read_csv("/home/achivuku/PycharmProjects/financedataanalysis/pricesvolumes.csv")
cols = [1,2,3,4,6,8,10,12,14,16,18,20,21,22,23,24,26,28,30,32,33,34,36,38,40,42]

df.drop(df.columns[cols],axis=1,inplace=True)
df.fillna(0, inplace=True)

allcols = df.columns.tolist()


allcols.remove("Date")
allcols.remove("IXIC_prices")
allcols.remove("B_prices")
allcols.remove("LAKE_prices")
allcols.remove("SUN_prices")


# print(allcols)
# sys.exit()



grumodelrmse = {}
GrumodeloutputPath = "/data/achivuku/PycharmProjects/financedataanalysis/grumodelrmse.pkl"




# for stock in allcols[1:]:
# 	# stock = allcols[2]
#
# 	stock_data = df[['Date', stock]]
# 	stock_data = stock_data.set_index('Date')
#
# 	raw_values = stock_data[[stock]].values.flatten()
# 	# diff_values = difference(raw_values, 1)
# 	# supervised = timeseries_to_supervised(diff_values, 1)
#
#
# 	n_lag = 200
# 	n_seq = 1
# 	nb_epoch = 10
# 	n_neurons = 200
# 	diff_series = difference(raw_values, 1)
# 	diff_values = diff_series.values
# 	diff_values = diff_values.reshape(len(diff_values), 1)
# 	scaler = MinMaxScaler(feature_range=(-1, 1))
# 	scaled_values = scaler.fit_transform(diff_values)
# 	scaled_values = scaled_values.reshape(len(scaled_values), 1)
# 	supervised = series_to_supervised(scaled_values, n_lag, n_seq)
# 	supervised_values = supervised.values
#
#
# 	numrecords = len(supervised_values)
# 	numtrainrecords = int(math.ceil(0.7*numrecords))
# 	numtestrecords = int(math.ceil(0.3*numrecords))
#
# 	train, test = supervised_values[:numtrainrecords], supervised_values[-numtestrecords:]
# 	model = fit_gru(train, n_lag, n_seq, 1, nb_epoch, n_neurons)
#
# 	# make forecasts
# 	forecasts = make_forecasts(model, 1, train, test, n_lag, n_seq)
# 	forecasts = inverse_transform(Series(raw_values), forecasts, scaler, numtestrecords+2)
#
# 	actual = [row[n_lag:] for row in test]
# 	actual = inverse_transform(Series(raw_values), actual, scaler, numtestrecords+2)
#
# 	rmsep = evaluate_forecasts(actual, forecasts, n_lag, n_seq)
# 	grumodelrmse[stock] = rmsep
#
# print(grumodelrmse)
# with open(GrumodeloutputPath, 'wb') as handle:
#     pickle.dump(grumodelrmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
# {'B_prices': 0.5632371691798901, 'LAKE_prices': 0.5010597957943063, 'AAPL_prices': 1.4753040684274836, 'APA_prices': 1.5203040212874435, 'SUN_prices': 3.8690563593132525, 'ABT_prices': 0.46921741739352213, 'WWD_prices': 0.8198175288199699, 'AEM_prices': 1.1078690573531664, 'T_prices': 0.3390494213141329, 'UTX_prices': 1.113961551744311, 'AFG_prices': 0.5886891985232372, 'MSFT_prices': 0.6060960139793086, 'ORCL_prices': 0.5207880938477653, 'MCD_prices': 0.9947724676998279, 'IXIC_prices': 40.78502876092441, 'CAT_prices': 1.4528155066700854}

# print(train)
# print(train.shape)
# print(test)
# print(test.shape)
# sys.exit()



# print('Calling fit_lstm')
# scaler, train_scaled, test_scaled = scale(train, test)
# lstm_model = fit_lstm(train_scaled, 1, 15, 10)
# train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
# lstm_model.predict(train_reshaped, batch_size=1)
# predictions = list()
# for i in range(len(test_scaled)):
#     X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
#     yhat = forecast_lstm(lstm_model, 1, X)
#     yhat = invert_scale(scaler, X, yhat)
#     yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
#     predictions.append(yhat)
#     expected = raw_values[len(train) + i]
#     print('Predicted=%f, Expected=%f' % (yhat, expected))
# mse = mse(predictions,raw_values[-numtestrecords:])
# print('Test MSE: %.3f' % mse)


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













# def preprocessinput(scaler, stock_data, stock, n_lag,n_seq,nb_epoch,n_neurons):
# 	# stock_data = stock_data.set_index('Date')
#
# 	raw_values = stock_data[[stock]].values.flatten()
#
# 	diff_series = difference(raw_values, 1)
# 	diff_values = diff_series.values
# 	diff_values = diff_values.reshape(len(diff_values), 1)
# 	scaled_values = scaler.fit_transform(diff_values)
# 	scaled_values = scaled_values.reshape(len(scaled_values), 1)
# 	supervised = series_to_supervised(scaled_values, n_lag, n_seq)
# 	supervised_values = supervised.values
# 	return (supervised_values, raw_values)
#




# n_lag = 200
# n_seq = 1
# nb_epoch = 15
# # nb_epoch = 1
# n_neurons = 25
# scaler = MinMaxScaler(feature_range=(-1, 1))
#
# grurnnmodelrrmse = collections.defaultdict(dict)
# grurnnmodelurrmse = collections.defaultdict(dict)
#
# GrurnnmodelroutputPath = "/data/achivuku/PycharmProjects/financedataanalysis/grurnnmodelrrmse.pkl"
# GrurnnmodeluroutputPath = "/data/achivuku/PycharmProjects/financedataanalysis/grurnnmodelurrmse.pkl"
#
# for stock1,stock2 in itertools.permutations(allcols,2):
#     print('stock1,stock2',stock1,stock2)
#
#     supervised1_values, raw1_values  = preprocessinput(scaler, df[['Date', stock1]],stock1,n_lag,n_seq,nb_epoch,n_neurons)
#
#     # print(supervised1_values[:, n_lag:].shape)
#     # sys.exit()
#
#     numrecordsr = len(supervised1_values)
#     numtrainrecordsr = int(math.ceil(0.7 * numrecordsr))
#     numtestrecordsr = int(math.ceil(0.3 * numrecordsr))
#
#     trainr, testr = supervised1_values[:numtrainrecordsr], supervised1_values[-numtestrecordsr:]
#     modelr = fit_gru(trainr, n_lag, n_seq, 1, nb_epoch, n_neurons)
#
#     forecastsr = make_forecasts(modelr, 1, trainr, testr, n_lag, n_seq)
#     forecastsr = inverse_transform(Series(raw1_values), forecastsr, scaler, numtestrecordsr + 2)
#
#     actualr = [row[n_lag:] for row in testr]
#     actualr = inverse_transform(Series(raw1_values), actualr, scaler, numtestrecordsr + 2)
#
#     rmser = evaluate_forecasts(actualr, forecastsr, n_lag, n_seq)
#
#     grurnnmodelrrmse[stock1][stock2] = rmser
#
#
#
#
#     supervised2_values, raw2_values = preprocessinput(scaler, df[['Date', stock2]],stock2,n_lag,n_seq,nb_epoch,n_neurons)
#     supervised2_values[:, n_lag:] = supervised1_values[:, n_lag:]
#     raw2_values = raw1_values
#
#     supervised_values = np.concatenate((supervised1_values, supervised2_values))
#
#     # supervised_values = pd.concat([supervised1_values,supervised2_values],ignore_index=True)
#
#     raw_values = np.concatenate((raw1_values, raw2_values))
#
#     numrecords = len(supervised_values)
#     numtrainrecords = int(math.ceil(0.7 * numrecords))
#     numtestrecords = int(math.ceil(0.3 * numrecords))
#
#     train, test = supervised_values[:numtrainrecords], supervised_values[-numtestrecords:]
#
#     modelur = fit_gru(train, n_lag, n_seq, 1, nb_epoch, n_neurons)
#
#     forecastsur = make_forecasts(modelur, 1, train, test, n_lag, n_seq)
#     forecastsur = inverse_transform(Series(raw_values), forecastsur, scaler, numtestrecords + 2)
#
#     actualur = [row[n_lag:] for row in test]
#     actualur = inverse_transform(Series(raw_values), actualur, scaler, numtestrecords + 2)
#
#     rmseur = evaluate_forecasts(actualur, forecastsur, n_lag, n_seq)
#     grurnnmodelurrmse[stock1][stock2] = rmseur
#
#
#
#
#
#
# print('grurnnmodelrrmse',grurnnmodelrrmse)
# with open(GrurnnmodelroutputPath, 'wb') as handle:
#     pickle.dump(grurnnmodelrrmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# print('grurnnmodelurrmse',grurnnmodelurrmse)
# with open(GrurnnmodeluroutputPath, 'wb') as handle:
#     pickle.dump(grurnnmodelurrmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#
# sys.exit()

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


def preprocess(datacol,scaler):
    raw_values = datacol.values.flatten()

    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values), 1)
    scaled_values = scaler.fit_transform(diff_values).flatten()
    # scaled_values = scaled_values.reshape(len(scaled_values), 1)
    return pd.Series(scaled_values)


def series_to_supervised(df, n_in=1):
    n_vars = 1 if type(df) is list else df.shape[1]
    cols = []
    names = []
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    Ypast = concat(cols, axis=1)

    Ypast.columns = names
    # Ypast.dropna(inplace=True)
    Ypast.fillna(0, inplace=True)
    # print('Ypast',Ypast.isnull().sum().sum())
    # print('Ypast.shape Here',Ypast.shape)
    # sys.exit()
    return Ypast


def fit_rnn(X, y, n_lag, n_batch, nb_epoch, n_neurons, networktype):
    X = X.reshape(X.shape[0], 1, X.shape[1])

    model = Sequential()

    if(networktype=="GRU"):
        model.add(GRU(2 * n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True, return_sequences=True))
        model.add(Dropout(0.5))
        # model.add(LSTM(n_neurons))
        model.add(GRU(n_neurons, stateful=True, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(GRU(n_neurons, stateful=True, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(n_neurons))
        model.add(Dropout(0.5))
        model.add(Dense(1))
    if (networktype == "LSTM"):
        model.add(LSTM(2 * n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(n_neurons))
        model.add(Dropout(0.5))
        model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
        model.reset_states()
    return model

def inverse_difference(last_ob, forecast):
	# invert first forecast
	inverted = list()
	inverted.append(forecast[0] + last_ob)
	# propagate difference forecast using inverted first value
	for i in range(1, len(forecast)):
		inverted.append(forecast[i] + inverted[i-1])
	return inverted

def inverse_transform(series, forecasts, scaler, n_test):
	inverted = list()
	for i in range(len(forecasts)):
		# # create array from forecast
		# forecast = np.array(forecasts[i])
		# forecast = forecast.reshape(1, len(forecast))
		# invert scaling
		inv_scale = scaler.inverse_transform(forecasts[i])
		inv_scale = inv_scale[0, :]
		# invert differencing
		index = len(series) - n_test + i - 1
		last_ob = series.values[index]
		inv_diff = inverse_difference(last_ob, inv_scale)
		# store
		inverted.append(inv_diff)
	return inverted

def forecast_gru(model, X, n_batch):
	# reshape input pattern to [samples, timesteps, features]
	X = X.reshape(1, 1, len(X))
	forecast = model.predict(X, batch_size=n_batch)
	return forecast[0,0]

# evaluate the persistence model
def make_forecasts(model, n_batch, test):
	forecasts = list()
	for i in range(len(test)):
		forecast = forecast_gru(model, test[i,:], n_batch)
		forecasts.append(forecast)
	return forecasts


GrurnnmodelurrmsegraphPath = "/data/achivuku/PycharmProjects/financedataanalysis/grurnnmodelurrmsegraph.pkl"
GrurnnmodeluroutputPath = "/data/achivuku/PycharmProjects/financedataanalysis/grurnnmodelurrmse.pkl"

handle = open(GrurnnmodeluroutputPath, 'rb')
grurnnmodelurrmse = pickle.load(handle)
grurnnmodelgraph = nx.read_gpickle(GrurnnmodelurrmsegraphPath)
# print('grurnnmodelurrmse', grurnnmodelurrmse.keys())

effects = []
for effect in sorted(grurnnmodelgraph.nodes()):
    predecessors = list(grurnnmodelgraph.predecessors(effect))
    if(predecessors):
        effects.append(effect + '_prices')
    # print(list(grurnnmodelgraph.predecessors(effect)))
# print('effects',sorted(effects))



# sys.exit()
# effects1 = ['AAPL_prices']
# effects2 = ['AEM_prices']
# effects3 = ['AFG_prices']
effects4 = ['APA_prices']
# effects5 = ['CAT_prices']
# effects6 = ['MCD_prices']
# effects7 = ['MSFT_prices']
# effects8 = ['ORCL_prices']
# effects9 = ['T_prices']
# effects10 = ['UTX_prices']
# effects11 = ['WWD_prices']

grurnnmodelmultiurrmse = collections.defaultdict(dict)
grurnnmodelmultiurfstat = collections.defaultdict(dict)

n_lag = 200
nb_epoch = 30
n_neurons = 500
scaler = MinMaxScaler(feature_range=(-1, 1))

# for effect in effects: # Split effects list for parallel processing on 4 servers
# for effect in effects1:
# for effect in effects2:
# for effect in effects3:
for effect in effects4:

# for effect in effects5:
# for effect in effects6:
# for effect in effects7:
# for effect in effects8:

# for effect in effects9:
# for effect in effects10:
# for effect in effects11:
# for effect in effects12:

    onecauses = zip([col + '_prices' for col in grurnnmodelgraph.predecessors(effect.rstrip('_prices'))])


    if onecauses:
        maximalcauses = onecauses
        firstiter = 0

        while maximalcauses:
            firstiter += 1

            candidatecauseslist = list(prod for prod in itertools.product(onecauses, maximalcauses) if prod[0][0] not in prod[1])

            print('candidatecauseslist',candidatecauseslist)
            print('onecauses',onecauses)
            print('maximalcauses',maximalcauses)

            # sys.exit()
            maximalcauses = []
            for candidatecauses in candidatecauseslist:
                stockcauses = []
                for c in candidatecauses:
                    for uc in c:
                        stockcauses.append(uc)
                stockcauses = filter(None, stockcauses)
                print('stockcauses',stockcauses)

                if(firstiter == 1) :
                    restrictederror = grurnnmodelurrmse[effect][stockcauses[0]]
                else:
                    restrictederror = grurnnmodelmultiurrmse[effect][tuple(sorted(stockcauses[1:]))]


                print('restrictederror',restrictederror)


                raw_values = df[effect]


                cols = []
                for col in df[stockcauses]:
                    cols.append(preprocess(df[col],scaler))



                indata = series_to_supervised(concat(cols, axis=1), n_lag) # Comment the onecauses series_to_supervised
                Ypast = indata.values
                outdata = preprocess(df[effect], scaler)
                Ycurr = outdata.values

                print('Ypast',Ypast)
                print('Ycurr',Ycurr)
                print('Ypast.shape',Ypast.shape)
                print('Ycurr.shape',Ycurr.shape)


                numrecords = len(Ycurr)
                numtestrecords = int(math.ceil(0.3 * numrecords))
                numtrainrecords = int(math.ceil(0.7 * numrecords))

                modelur = fit_rnn(Ypast[:numtrainrecords], Ycurr[:numtrainrecords], n_lag, 1, nb_epoch, n_neurons, "GRU")
                forecasts = make_forecasts(modelur, 1, Ypast[-numtestrecords:])

                Ycurrp = inverse_transform(Series(raw_values), forecasts, scaler, numtestrecords + 2)

                unrestrictederror = math.sqrt(mean_squared_error(Ycurrp, Ycurr[-numtestrecords:]))

                fstat = (restrictederror - unrestrictederror) / unrestrictederror

                print('unrestrictederror',unrestrictederror)
                print('fstat',fstat)



                if ( fstat > 0.05 ):
                    founddup = False
                    for maximalcause in maximalcauses:
                        if sorted(list(maximalcause)) == sorted(stockcauses):
                            founddup = True

                    print('stockcauses before',stockcauses)
                    print('maximalcauses before',maximalcauses)
                    print('founddup before',founddup)

                    if(not founddup):
                        maximalcauses.append(tuple(sorted(stockcauses)))
                        grurnnmodelmultiurrmse[effect][tuple(sorted(stockcauses))] = unrestrictederror
                        grurnnmodelmultiurfstat[effect][tuple(sorted(stockcauses))] = fstat
                        print('maximalcauses after',maximalcauses)

print('maximalcauses',maximalcauses)
print('grurnnmodelmultiurrmse',grurnnmodelmultiurrmse)
print('grurnnmodelmultiurfstat',grurnnmodelmultiurfstat)

if(maximalcauses):
    # GrurnnmodelmultiurrmsePath = "/data/achivuku/PycharmProjects/financedataanalysis/grurnnmodelmultiurrmse1.pkl"
    # GrurnnmodelmultiurfstatPath = "/data/achivuku/PycharmProjects/financedataanalysis/grurnnmodelmultiurfstat1.pkl"

    # GrurnnmodelmultiurrmsePath = "/data/achivuku/PycharmProjects/financedataanalysis/grurnnmodelmultiurrmse2.pkl"
    # GrurnnmodelmultiurfstatPath = "/data/achivuku/PycharmProjects/financedataanalysis/grurnnmodelmultiurfstat2.pkl"

    # GrurnnmodelmultiurrmsePath = "/data/achivuku/PycharmProjects/financedataanalysis/grurnnmodelmultiurrmse3.pkl"
    # GrurnnmodelmultiurfstatPath = "/data/achivuku/PycharmProjects/financedataanalysis/grurnnmodelmultiurfstat3.pkl"

    # GrurnnmodelmultiurrmsePath = "/data/achivuku/PycharmProjects/financedataanalysis/grurnnmodelmultiurrmse4.pkl"
    # GrurnnmodelmultiurfstatPath = "/data/achivuku/PycharmProjects/financedataanalysis/grurnnmodelmultiurfstat4.pkl"
    #
    #
    #
    #
    #
    #
    #
    # GrurnnmodelmultiurrmsePath = "/data/achivuku/PycharmProjects/financedataanalysis/grurnnmodelmultiurrmse5.pkl"
    # GrurnnmodelmultiurfstatPath = "/data/achivuku/PycharmProjects/financedataanalysis/grurnnmodelmultiurfstat5.pkl"

    # GrurnnmodelmultiurrmsePath = "/data/achivuku/PycharmProjects/financedataanalysis/grurnnmodelmultiurrmse6.pkl"
    # GrurnnmodelmultiurfstatPath = "/data/achivuku/PycharmProjects/financedataanalysis/grurnnmodelmultiurfstat6.pkl"

    # GrurnnmodelmultiurrmsePath = "/data/achivuku/PycharmProjects/financedataanalysis/grurnnmodelmultiurrmse7.pkl"
    # GrurnnmodelmultiurfstatPath = "/data/achivuku/PycharmProjects/financedataanalysis/grurnnmodelmultiurfstat7.pkl"

    # GrurnnmodelmultiurrmsePath = "/data/achivuku/PycharmProjects/financedataanalysis/grurnnmodelmultiurrmse8.pkl"
    # GrurnnmodelmultiurfstatPath = "/data/achivuku/PycharmProjects/financedataanalysis/grurnnmodelmultiurfstat8.pkl"
    #
    #
    #
    #
    #
    #
    # GrurnnmodelmultiurrmsePath = "/data/achivuku/PycharmProjects/financedataanalysis/grurnnmodelmultiurrmse9.pkl"
    # GrurnnmodelmultiurfstatPath = "/data/achivuku/PycharmProjects/financedataanalysis/grurnnmodelmultiurfstat9.pkl"
    #
    # GrurnnmodelmultiurrmsePath = "/data/achivuku/PycharmProjects/financedataanalysis/grurnnmodelmultiurrmse10.pkl"
    # GrurnnmodelmultiurfstatPath = "/data/achivuku/PycharmProjects/financedataanalysis/grurnnmodelmultiurfstat10.pkl"
    #
    GrurnnmodelmultiurrmsePath = "/data/achivuku/PycharmProjects/financedataanalysis/grurnnmodelmultiurrmse11.pkl"
    GrurnnmodelmultiurfstatPath = "/data/achivuku/PycharmProjects/financedataanalysis/grurnnmodelmultiurfstat11.pkl"
    #
    # GrurnnmodelmultiurrmsePath = "/data/achivuku/PycharmProjects/financedataanalysis/grurnnmodelmultiurrmse12.pkl"
    # GrurnnmodelmultiurfstatPath = "/data/achivuku/PycharmProjects/financedataanalysis/grurnnmodelmultiurfstat12.pkl"


    with open(GrurnnmodelmultiurrmsePath, 'wb') as handle:
        pickle.dump(grurnnmodelmultiurrmse, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(GrurnnmodelmultiurfstatPath, 'wb') as handle:
        pickle.dump(grurnnmodelmultiurfstat, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # TO DO : After all runs, create only one GrurnnmodelmultiurrmsePath and GrurnnmodelmultiurfstatPath
