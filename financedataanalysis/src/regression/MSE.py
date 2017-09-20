import sys
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

df = pd.read_csv("/home/achivuku/Documents/financedataanalysis/pricesvolumes.csv")

cols = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,33,34,36,38,40,42]
df.drop(df.columns[cols],axis=1,inplace=True)
df['Date'] = pd.to_datetime(df['Date'])

print(len(df.columns)) # 21
print((df.columns)) # Index([u'Date', u'^DJI_prices', u'^GSPC_prices', u'^IXIC_prices', u'AAPL_prices', u'ABT_prices', u'AEM_prices', u'AFG_prices', u'APA_prices', u'B_prices', u'CAT_prices', u'FRD_prices', u'GIGA_prices', u'LAKE_prices', u'MCD_prices', u'MSFT_prices', u'ORCL_prices', u'SUN_prices', u'T_prices', u'UTX_prices', u'WWD_prices'], dtype='object')

# print(len(df.columns))
# print(df['Date'])

p = 100
q = 100
inputbatchsize = 5100

def getstockdata(dfone):
    Ypast = []
    Ycurr = []
    for i in xrange(-inputbatchsize, 0):
        y = dfone.iloc[i,1]
        x = dfone.iloc[i - p:i,1].tolist()
        Ypast.append(x)
        Ycurr.append(y)
    Ypast = np.vstack(Ypast)
    Ycurr = np.vstack(Ycurr)
    Ycurr = Ycurr.reshape(Ycurr.shape[0], )
    return Ypast,Ycurr

def restricted_model(lag):
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

Ypast,Ycurr = getstockdata(df[['Date','MSFT_prices']])
modelr = restricted_model(p)
np.random.seed(3)
modelr.fit(Ypast, Ycurr, epochs=5, batch_size=32, verbose=2, validation_split=0.1)
Ycurrp = modelr.predict(Ypast, batch_size=128)
mse_valuer, mae_valuer = modelr.evaluate(Ypast, Ycurr, batch_size=128, verbose=1)

print('\n')
print('modelr Ycurrp.mean()',Ycurrp.mean())
print('modelr Ycurrp.std()',Ycurrp.std())
print('modelr mse_value',mse_valuer)
print('modelr mae_value',mae_valuer)
print('modelr r2_score(Ycurrp, Ycurr)',r2_score(Ycurrp, Ycurr))

Ypast,Ycurr = getstockdata(df[['Date','ORCL_prices']])
modelur = restricted_model(q)
np.random.seed(7)
modelur.fit(Ypast, Ycurr, epochs=5, batch_size=32, verbose=2, validation_split=0.1)
Ycurrp = modelur.predict(Ypast, batch_size=128)
mse_valuer, mae_valuer = modelur.evaluate(Ypast, Ycurr, batch_size=128, verbose=1)

print('\n')
print('modelur Ycurrp.mean()',Ycurrp.mean())
print('modelur Ycurrp.std()',Ycurrp.std())
print('modelur mse_value',mse_valuer)
print('modelur mae_value',mae_valuer)
print('modelur r2_score(Ycurrp, Ycurr)',r2_score(Ycurrp, Ycurr))

# https://github.com/fchollet/keras/tree/master/examples
# TO DO : Output causal graph for every pairs of stock prices