import sys
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

from keras import backend as K

import tensorflow as tf
import math
import itertools

import collections
import pickle

QmemodelrmaePath = "/home/achivuku/PycharmProjects/financedataanalysis/qmemodelrmae.pkl"
QmemodelrmsePath = "/home/achivuku/PycharmProjects/financedataanalysis/qmemodelrmse.pkl"
QmemodelurmaePath = "/home/achivuku/PycharmProjects/financedataanalysis/qmemodelurmae.pkl"
QmemodelurmsePath = "/home/achivuku/PycharmProjects/financedataanalysis/qmemodelurmse.pkl"

MsemodelrmaePath = "/home/achivuku/PycharmProjects/financedataanalysis/msemodelrmae.pkl"
MsemodelrmsePath = "/home/achivuku/PycharmProjects/financedataanalysis/msemodelrmse.pkl"
MsemodelurmaePath = "/home/achivuku/PycharmProjects/financedataanalysis/msemodelurmae.pkl"
MsemodelurmsePath = "/home/achivuku/PycharmProjects/financedataanalysis/msemodelurmse.pkl"



df = pd.read_csv("/home/achivuku/Documents/financedataanalysis/pricesvolumes.csv")
# print(df.columns)
# sys.exit()

# cols = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,33,34,36,38,40,42]
cols = [1,2,3,4,6,8,10,12,14,16,18,20,21,22,23,24,26,28,30,32,33,34,36,38,40,42]

df.drop(df.columns[cols],axis=1,inplace=True)
# df['Date'] = pd.to_datetime(df['Date'])

print(len(df.columns)) # 17

print((df.columns)) # Index([u'Date', u'^DJI_prices', u'^GSPC_prices', u'^IXIC_prices', u'AAPL_prices', u'ABT_prices', u'AEM_prices', u'AFG_prices', u'APA_prices', u'B_prices', u'CAT_prices', u'FRD_prices', u'GIGA_prices', u'LAKE_prices', u'MCD_prices', u'MSFT_prices', u'ORCL_prices', u'SUN_prices', u'T_prices', u'UTX_prices', u'WWD_prices'], dtype='object')
print(len(df.index)) # 5285

allcols = df.columns.tolist()
print('allcols',allcols[1:])
df[allcols[1:]] = df[allcols[1:]].apply(pd.to_numeric).apply(lambda x: x/x.mean(), axis=0)

allcols.remove("Date")
# allcols.remove("DJI_prices")
# allcols.remove("FRD_prices")
# allcols.remove("GSPC_prices")
# allcols.remove("GIGA_prices")


# print(len(df.columns))
# print(df['Date'])

inputbatchsize = 5000
p = 200
q = 200

percentilenum = 10
numepochs = 10

def getstockdata(dfone, lag):
    Ypast = []
    Ycurr = []
    for i in xrange(-inputbatchsize, 0):
        y = dfone.iloc[i,1]
        x = dfone.iloc[i - lag:i,1].tolist()
        Ypast.append(x)
        Ycurr.append(y)
    Ypast = np.vstack(Ypast)
    Ycurr = np.vstack(Ycurr)
    Ycurr = Ycurr.reshape(Ycurr.shape[0], )
    return Ypast,Ycurr

def restricted_mse_model(lag):
    model = Sequential()
    model.add(Dense(units=2*lag, activation='relu', kernel_initializer='normal', bias_initializer='zeros', input_dim=Ypast.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(lag/2, activation='linear', kernel_initializer='normal', bias_initializer='zeros'))
    model.add(Dropout(0.2))
    model.add(Dense(lag/2, activation='relu', kernel_initializer='normal', bias_initializer='zeros'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, kernel_initializer='normal', bias_initializer='zeros'))

    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # model.compile(loss='mse', optimizer='adam', metrics=['msd'])
    return model

def quadratic_mean_error(y_true, y_pred):

    sumofsquares = 0
    currpercentile = 0
    prevpercentile = 0
    for i in xrange(10, 110, percentilenum):
        prevpercentile = currpercentile
        currpercentile = tf.contrib.distributions.percentile(y_true, q=i)
        booleaninterpercentile = tf.logical_and(tf.less(y_true,currpercentile),tf.greater(y_true,prevpercentile))
        trueslice = tf.boolean_mask(y_true, booleaninterpercentile)
        predslice = tf.boolean_mask(y_pred, booleaninterpercentile)
        sumofsquares += tf.to_float(K.square(K.mean(K.square(predslice - trueslice), axis=-1)))
    return K.sqrt(sumofsquares/10)


    # percentile1 = tf.contrib.distributions.percentile(y_true, q=10.)
    # percentile2 = tf.contrib.distributions.percentile(y_true, q=20.)
    # percentile3 = tf.contrib.distributions.percentile(y_true, q=30.)
    # percentile4 = tf.contrib.distributions.percentile(y_true, q=40.)
    # percentile5 = tf.contrib.distributions.percentile(y_true, q=50.)
    # percentile6 = tf.contrib.distributions.percentile(y_true, q=60.)
    # percentile7 = tf.contrib.distributions.percentile(y_true, q=70.)
    # percentile8 = tf.contrib.distributions.percentile(y_true, q=80.)
    # percentile9 = tf.contrib.distributions.percentile(y_true, q=90.)
    # percentile10 = tf.contrib.distributions.percentile(y_true, q=100.)
    #
    # booleaninterpercentile1 = tf.logical_and(tf.less(y_true,percentile1),tf.greater(y_true,0))
    # booleaninterpercentile2 = tf.logical_and(tf.less(y_true,percentile2),tf.greater(y_true,percentile1))
    # booleaninterpercentile3 = tf.logical_and(tf.less(y_true,percentile3),tf.greater(y_true,percentile2))
    # booleaninterpercentile4 = tf.logical_and(tf.less(y_true,percentile4),tf.greater(y_true,percentile3))
    # booleaninterpercentile5 = tf.logical_and(tf.less(y_true,percentile5),tf.greater(y_true,percentile4))
    # booleaninterpercentile6 = tf.logical_and(tf.less(y_true,percentile7),tf.greater(y_true,percentile6))
    # booleaninterpercentile7 = tf.logical_and(tf.less(y_true,percentile8),tf.greater(y_true,percentile7))
    # booleaninterpercentile8 = tf.logical_and(tf.less(y_true,percentile9),tf.greater(y_true,percentile8))
    # booleaninterpercentile9 = tf.logical_and(tf.less(y_true,percentile10),tf.greater(y_true,percentile9))
    #
    # trueslice1 = tf.boolean_mask(y_true,booleaninterpercentile1)
    # predslice1 = tf.boolean_mask(y_pred,booleaninterpercentile1)
    #
    # trueslice2 = tf.boolean_mask(y_true,booleaninterpercentile2)
    # predslice2 = tf.boolean_mask(y_pred,booleaninterpercentile2)
    #
    # trueslice3 = tf.boolean_mask(y_true,booleaninterpercentile3)
    # predslice3 = tf.boolean_mask(y_pred,booleaninterpercentile3)
    #
    # trueslice4 = tf.boolean_mask(y_true,booleaninterpercentile4)
    # predslice4 = tf.boolean_mask(y_pred,booleaninterpercentile4)
    #
    # trueslice5 = tf.boolean_mask(y_true,booleaninterpercentile5)
    # predslice5 = tf.boolean_mask(y_pred,booleaninterpercentile5)
    #
    # trueslice6 = tf.boolean_mask(y_true,booleaninterpercentile6)
    # predslice6 = tf.boolean_mask(y_pred,booleaninterpercentile6)
    #
    # trueslice7 = tf.boolean_mask(y_true,booleaninterpercentile7)
    # predslice7 = tf.boolean_mask(y_pred,booleaninterpercentile7)
    #
    # trueslice8 = tf.boolean_mask(y_true,booleaninterpercentile8)
    # predslice8 = tf.boolean_mask(y_pred,booleaninterpercentile8)
    #
    # trueslice9 = tf.boolean_mask(y_true,booleaninterpercentile9)
    # predslice9 = tf.boolean_mask(y_pred,booleaninterpercentile9)
    #
    # return K.sqrt(K.mean(K.mean(K.square(predslice1 - trueslice1), axis=-1) + K.mean(K.square(predslice2 - trueslice2), axis=-1)) + K.mean(K.mean(K.square(predslice3 - trueslice3), axis=-1) + K.mean(K.square(predslice4 - trueslice4), axis=-1)) + K.mean(K.mean(K.square(predslice5 - trueslice5), axis=-1) + K.mean(K.square(predslice6 - trueslice6), axis=-1)) + K.mean(K.mean(K.square(predslice7 - trueslice7), axis=-1) + K.mean(K.square(predslice8 - trueslice8), axis=-1)) + K.mean(K.square(predslice9 - trueslice9), axis=-1))
    # return K.sqrt(K.mean(K.mean(K.square(predslice1 - trueslice1), axis=-1) + K.mean(K.square(predslice2 - trueslice2), axis=-1)))

def restricted_qme_model(lag):
    model = Sequential()
    model.add(Dense(units=2*lag, activation='relu', kernel_initializer='normal', bias_initializer='zeros', input_dim=Ypast.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(lag/2, activation='linear', kernel_initializer='normal', bias_initializer='zeros'))
    model.add(Dropout(0.2))
    model.add(Dense(lag/2, activation='relu', kernel_initializer='normal', bias_initializer='zeros'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, kernel_initializer='normal', bias_initializer='zeros'))

    model.compile(optimizer='adam', loss=quadratic_mean_error, metrics=['mae'])
    return model

# qmemodelrmae = collections.defaultdict(dict)
# qmemodelrmse = collections.defaultdict(dict)
# qmemodelurmae = collections.defaultdict(dict)
# qmemodelurmse = collections.defaultdict(dict)
#
# # for stock1,stock2 in itertools.combinations(allcols,2):
# #     qmemodelrmae[stock1][stock2] = 0
# #     qmemodelrmse[stock1][stock2] = 0
# #
# #     qmemodelurmae[stock1][stock2] = 0
# #     qmemodelurmse[stock1][stock2] = 0
#
# for stock1,stock2 in itertools.combinations(allcols,2):
#     print('stock1,stock2',stock1,stock2)
#     Ypast, Ycurr = getstockdata(df[['Date', stock1]], p)
#     # Ypast, Ycurr = getstockdata(df[['Date', 'MSFT_prices']])
#     numrecords = len(Ycurr)
#     numtestrecords = int(math.ceil(0.3*numrecords))
#     numtrainrecords = int(math.ceil(0.7*numrecords))
#
#     modelr = restricted_qme_model(p)
#     np.random.seed(3)
#     modelr.fit(Ypast[:numtrainrecords], Ycurr[:numtrainrecords], epochs=numepochs, batch_size=32, verbose=2, validation_split=0.1)
#     Ycurrp = modelr.predict(Ypast[-numtestrecords:], batch_size=128)
#     qme_mse_valuer, qme_mae_valuer = modelr.evaluate(Ypast[-numtestrecords:], Ycurr[-numtestrecords:], batch_size=128, verbose=1)
#
#     print('\n')
#     print('qme modelr Ycurrp.mean()',Ycurrp.mean())
#     print('qme modelr Ycurrp.std()',Ycurrp.std())
#     print('qme modelr mae_value',qme_mae_valuer)
#     print('qme modelr mse_value',qme_mse_valuer)
#     print('qme modelr r2_score(Ycurrp, Ycurr)',r2_score(Ycurrp, Ycurr[-numtestrecords:]))
#
#     qmemodelrmae[stock1][stock2] = qme_mae_valuer
#     qmemodelrmse[stock1][stock2] = qme_mse_valuer
#
#     Ypast1, Ycurr1 = getstockdata(df[['Date', stock1]], p)
#     Ypast2, Ycurr2 = getstockdata(df[['Date', stock2]], q)
#     Ycurr2 = Ycurr1
#
#     Ypast = np.concatenate((Ypast1, Ypast2))
#     Ycurr = np.concatenate((Ycurr1, Ycurr2))
#
#     # Ypast,Ycurr = getstockdata(df[['Date', stock2]])
#     # Ypast,Ycurr = getstockdata(df[['Date','ORCL_prices']])
#     numrecords = len(Ycurr)
#     numtestrecords = int(math.ceil(0.3*numrecords))
#     numtrainrecords = int(math.ceil(0.7*numrecords))
#
#     modelur = restricted_qme_model(q)
#     np.random.seed(7)
#     modelur.fit(Ypast[:numtrainrecords], Ycurr[:numtrainrecords], epochs=numepochs, batch_size=32, verbose=2, validation_split=0.1)
#     Ycurrp = modelur.predict(Ypast[-numtestrecords:], batch_size=128)
#     qme_mse_valueur, qme_mae_valueur = modelur.evaluate(Ypast[-numtestrecords:], Ycurr[-numtestrecords:], batch_size=128, verbose=1)
#
#     print('\n')
#     print('qme modelur Ycurrp.mean()',Ycurrp.mean())
#     print('qme modelur Ycurrp.std()',Ycurrp.std())
#     print('qme modelur mae_value',qme_mae_valueur)
#     print('qme modelur mse_value',qme_mse_valueur)
#     print('qme modelur r2_score(Ycurrp, Ycurr)',r2_score(Ycurrp, Ycurr[-numtestrecords:]))
#
#     qmemodelurmae[stock1][stock2] = qme_mae_valueur
#     qmemodelurmse[stock1][stock2] = qme_mse_valueur
#
#
# print('qmemodelrmae',qmemodelrmae)
# print('qmemodelrmse',qmemodelrmse)
# print('qmemodelurmae',qmemodelurmae)
# print('qmemodelurmse',qmemodelurmse)
#
# with open(QmemodelrmaePath, 'wb') as handle:
#     pickle.dump(qmemodelrmae, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open(QmemodelrmsePath, 'wb') as handle:
#     pickle.dump(qmemodelrmse, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open(QmemodelurmaePath, 'wb') as handle:
#     pickle.dump(qmemodelurmae, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open(QmemodelurmsePath, 'wb') as handle:
#     pickle.dump(qmemodelurmse, handle, protocol=pickle.HIGHEST_PROTOCOL)


msemodelrmae = collections.defaultdict(dict)
msemodelrmse = collections.defaultdict(dict)
msemodelurmae = collections.defaultdict(dict)
msemodelurmse = collections.defaultdict(dict)

for stock1,stock2 in itertools.combinations(allcols,2):
    print('stock1,stock2',stock1,stock2)
    Ypast, Ycurr = getstockdata(df[['Date', stock1]],p)
    # Ypast,Ycurr = getstockdata(df[['Date','MSFT_prices']])
    numrecords = len(Ycurr)
    numtestrecords = int(math.ceil(0.3*numrecords))
    numtrainrecords = int(math.ceil(0.7*numrecords))

    modelr = restricted_mse_model(p)
    np.random.seed(3)
    modelr.fit(Ypast[:numtrainrecords], Ycurr[:numtrainrecords], epochs=numepochs, batch_size=32, verbose=2, validation_split=0.1)
    Ycurrp = modelr.predict(Ypast[-numtestrecords:], batch_size=128)
    mse_mse_valuer, mse_mae_valuer = modelr.evaluate(Ypast[-numtestrecords:], Ycurr[-numtestrecords:], batch_size=128, verbose=1)

    print('\n')
    print('mse modelr Ycurrp.mean()',Ycurrp.mean())
    print('mse modelr Ycurrp.std()',Ycurrp.std())
    print('mse modelr mae_value',mse_mae_valuer)
    print('mse modelr mse_value',mse_mse_valuer)
    print('mse modelr r2_score(Ycurrp, Ycurr)',r2_score(Ycurrp, Ycurr[-numtestrecords:]))

    msemodelrmae[stock1][stock2] = mse_mae_valuer
    msemodelrmse[stock1][stock2] = mse_mse_valuer

    Ypast1, Ycurr1 = getstockdata(df[['Date', stock1]],p)
    Ypast2, Ycurr2 = getstockdata(df[['Date', stock2]],q)
    Ycurr2 = Ycurr1

    Ypast = np.concatenate((Ypast1, Ypast2))
    Ycurr = np.concatenate((Ycurr1, Ycurr2))

    # Ypast, Ycurr = getstockdata(df[['Date', stock2]])
    # Ypast,Ycurr = getstockdata(df[['Date','ORCL_prices']])
    numrecords = len(Ycurr)
    numtestrecords = int(math.ceil(0.3*numrecords))
    numtrainrecords = int(math.ceil(0.7*numrecords))

    modelur = restricted_mse_model(q)
    np.random.seed(7)
    modelur.fit(Ypast[:numtrainrecords], Ycurr[:numtrainrecords], epochs=numepochs, batch_size=32, verbose=2, validation_split=0.1)
    Ycurrp = modelur.predict(Ypast[-numtestrecords:], batch_size=128)
    mse_mse_valueur, mse_mae_valueur = modelur.evaluate(Ypast[-numtestrecords:], Ycurr[-numtestrecords:], batch_size=128, verbose=1)

    print('\n')
    print('mse modelur Ycurrp.mean()',Ycurrp.mean())
    print('mse modelur Ycurrp.std()',Ycurrp.std())
    print('mse modelur mae_value',mse_mae_valueur)
    print('mse modelur mse_value',mse_mse_valueur)
    print('mse modelur r2_score(Ycurrp, Ycurr)',r2_score(Ycurrp, Ycurr[-numtestrecords:]))

    msemodelurmae[stock1][stock2] = mse_mae_valueur
    msemodelurmse[stock1][stock2] = mse_mse_valueur

print('msemodelrmae', msemodelrmae)
print('msemodelrmse', msemodelrmse)
print('msemodelurmae', msemodelurmae)
print('msemodelurmse', msemodelurmse)


with open(MsemodelrmaePath, 'wb') as handle:
    pickle.dump(msemodelrmae, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(MsemodelrmsePath, 'wb') as handle:
    pickle.dump(msemodelrmse, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(MsemodelurmaePath, 'wb') as handle:
    pickle.dump(msemodelurmae, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(MsemodelurmsePath, 'wb') as handle:
    pickle.dump(msemodelurmse, handle, protocol=pickle.HIGHEST_PROTOCOL)



# # https://github.com/fchollet/keras/tree/master/examples
# # TO DO : Output causal graph for every pairs of stock prices
# From this example and other examples of loss functions and metrics, the approach is to use standard math functions on the backend to calculate the metric of interest.
# You can't get the values from the tensor symbolic variable directly. Yo need to write a tensorflow function to extract the value.
# https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/
# https://github.com/fchollet/keras/blob/master/keras/backend/tensorflow_backend.py
