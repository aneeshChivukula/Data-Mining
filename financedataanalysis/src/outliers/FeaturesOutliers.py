import sys
import pandas as pd
import numpy as np
from itertools import chain
from sklearn.neighbors import LocalOutlierFactor
import math

def getlofindex(stock, lofw):
    lofindex = [0] * (len(stock))
    for i in xrange(0, len(stock) - 1, lofw):
        for j in xrange(0, lofw, 1):
            if (i + j < len(stock)):
                lofindex[i + j] = j + 1
    return lofindex

tw = 100
bw = 500
lofw1 = 30
lofw2 = 7
# Time windows to discretize daily closing prices
ndp = 3
nstd = 2
numneighbours = 20
# Optional LOF Score settings are distance metrics : algorithm, metric, p and parallel processes : n_jobs
# Optionally try LOF Class method _local_reachability_density

df = pd.read_csv("/home/achivuku/Documents/financedataanalysis/pricesvolumes.csv")
cols = [1,2,3,4,6,8,10,12,14,16,18,20,21,22,23,24,26,28,30,32,33,34,36,38,40,42]

df.drop(df.columns[cols],axis=1,inplace=True)

# print(df.head(10))
# sys.exit()

stockname = 'ORCL_prices'
dfstockname = stockname.rstrip('_prices')

stock = df[[stockname]].values
# bw = len(stock) - tw
flatstock = [round(x,ndp) for x in list(chain.from_iterable(stock))]
stockdiff = [0] * (len(stock))
stocklocalstats = []
stockmeans = [0] * (len(stock))
stockdeviations = [0] * (len(stock))
stockthresholds = [0] * (len(stock))

lofw = lofw1
lofindex1 = getlofindex(stock, lofw)
lofw = lofw2
lofindex2 = getlofindex(stock, lofw)

for i in xrange(len(stock)-1, tw, -tw): # for many benchmarking window, replace all nan with 0
# for i in xrange(len(stock)-1, len(stock)-bw, -tw): # for one benchmarking window
    for j in xrange(i, i-tw, -1):
        stockdiff[j] = round((stock[j] - stock[i-tw-1])[0], ndp)
    # stockstats.append(stockdiff[i])
    stocklocalstats = stockdiff[i - tw + 1:i + 1]
    m = np.mean(np.abs(stocklocalstats))
    sd = np.std(np.abs(stocklocalstats))
    t = round(m + nstd * sd, ndp)
    for j in xrange(i, i - tw, -1):
        stockmeans[j] = round(m, ndp)
        stockdeviations[j] = round(sd, ndp)
        stockthresholds[j] = round(t, ndp)

# m = np.mean(np.abs(stockstats))
# sd = np.std(np.abs(stockstats))
# t = m + nstd * sd

# stockdiff = stockdiff[-bw:]
# m = np.mean(np.abs(stockdiff))
# sd = np.std(np.abs(stockdiff))
# t = m + nstd * sd

# print(stockdiff)
#
# print(t)
# print(m)
# print(sd)
# print('\n')



stockstatsdf = pd.DataFrame(
    {
        # dfstockname + '_prices': flatstock,
        # dfstockname + '_pricediffs': stockdiff,
        dfstockname + '_pricemeans': stockmeans,
        dfstockname + '_pricedeviations': stockdeviations,
        dfstockname + '_pricethresholds': stockthresholds,
        dfstockname + '_pricetimeindex_months': lofindex1,
        dfstockname + '_pricetimeindex_weeks': lofindex2
    }
)

print(stockstatsdf.tail(10))
# print(stockstatsdf[[dfstockname + '_pricediffs']])
# print(stockstatsdf[[dfstockname + '_pricemeans']])
# print(stockstatsdf[[dfstockname + '_pricedeviations']])
# print(stockstatsdf[[dfstockname + '_pricethresholds']])
# print(stockstatsdf[[dfstockname + '_pricetimeindex_months']])
# print(stockstatsdf[[dfstockname + '_pricetimeindex_weeks']])

X = stockstatsdf.values

print(X.shape)
numrecords = len(X)
numtrainrecords = int(math.ceil(0.7 * numrecords))
numtestrecords = int(math.ceil(0.3 * numrecords))

clf = LocalOutlierFactor(n_neighbors=numneighbours)
clf.fit(X[:numtrainrecords])

y_pred = clf._predict(X[-numtestrecords:])
# y_pred = clf.fit_predict(X[:numtrainrecords])
# Returns -1 for anomalies/outliers and 1 for inliers.
print(y_pred)
print(len(y_pred[y_pred == -1]))

pred_lof_scores = clf._decision_function(X[-numtestrecords:])
# Returns The opposite of the Local Outlier Factor of each input samples. The lower, the more abnormal.
print(pred_lof_scores)
print(len(pred_lof_scores))
# print(clf._decision_function(X[:numtrainrecords]))
# print(clf.negative_outlier_factor_)