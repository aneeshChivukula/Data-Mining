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

generatestatsdf = False
StockDataframePath = "/data/achivuku/financedataanalysis/AlertsBySecurity/SEC0000001.alerts.filtered.csv"
StockstatsDataframePath = "/data/achivuku/PycharmProjects/financedataanalysis/stockstats.pkl"

ndp = 3
nstd = 2
numneighbours = 35
outliers_fraction = 0.01
# Optional LOF Score settings are distance metrics : algorithm, metric, p and parallel processes : n_jobs
# Optionally try LOF Class method _local_reachability_density


if(generatestatsdf == True):
    df = pd.read_csv(StockDataframePath)
    df = df[['Trans.ID','price','Timestamp','binarylabels']]

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # print(df['Timestamp'].dtype)
    # print(df.tail(10))
    # print(df[df.binarylabels == "P"])



    rowscount = len(df['Timestamp'])
    stocklocalstats = []
    stockmeans = [0] * (rowscount)
    stockdeviations = [0] * (rowscount)
    stockthresholds = [0] * (rowscount)
    # lofdayindex = [0] * (rowscount)
    # lofhourindex = [0] * (rowscount)


    # for currindex, currdatetime in df['Timestamp'].iteritems():
    #
    #     lofdayindex[currindex] = currdatetime.day
    #     if(currdatetime.minute <= 30):
    #         lofhourindex[currindex] = 2 * currdatetime.hour - 1
    #     else:
    #         lofhourindex[currindex] = 2 * currdatetime.hour
    #
    # df['timeindex_days'] = lofdayindex
    # df['timeindex_hours'] = lofhourindex

    df['TimestampIndex'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index('TimestampIndex')

    # print(df.head(10))
    # sys.exit()

    tw = 10*60*1000
    bw = 30*7*60*60*1000
    nummn = 30
    starthr = 9
    endhr = 16
    startmin = 0
    endmin = 61
    # print(df['Timestamp'].min()) # 2012-01-02 09:25:00
    # print(df['Timestamp'].max()) # 2012-12-27 16:00:03.868000

    InitialDaysDFList = [group[1] for group in df.groupby([df.index.year,df.index.month,df.index.day])]

    # print(InitialDaysDFList[-1])
    # print(len(InitialDaysDFList))

    FinalDaysDFList = []


    for i in xrange(len(InitialDaysDFList)-1,30,-1):
        dfl = InitialDaysDFList[i]

        MinutesDFList = []

        for hr in xrange(starthr,endhr):
            if(hr==9):
                startmin = 20
                nummn = 40
                endmin = 60
            else:
                nummn = 30
                startmin = 0
                endmin = 60

            for mn in xrange(startmin,endmin,nummn):
                dfls = dfl.loc[(dfl.index.hour == hr) & (mn < dfl.index.minute) & (dfl.index.minute < mn + nummn)]

                if(hr==15 and mn==30):
                    dflsl = dfl.loc[dfl.index.hour == hr+1]
                    dfls = dfls.append(dflsl)

                m = round(dfls['price'].mean(), ndp)
                sd = round(dfls['price'].std(ddof=1), ndp)

                dfl30 = InitialDaysDFList[i - 30:i]
                dflbl = []
                for dflp in dfl30:
                    dflsp = dflp.loc[(dflp.index.hour == hr) & (mn < dflp.index.minute) & (dflp.index.minute < mn + nummn)]

                    if (hr == 15 and mn == 30):
                        dflpl = dflp.loc[dflp.index.hour == hr + 1]
                        dflsp = dflp.append(dflpl)

                dflbl.append(dflsp)

                dflb30 = pd.concat(dflbl)
                mb30 = round(dflb30['price'].mean(), ndp)
                sdb30 = round(dflb30['price'].std(ddof=1), ndp)

                mb1 = round(dfl['price'].mean(), ndp)
                sdb1 = round(dfl['price'].std(ddof=1), ndp)

                dflb7 = pd.concat(InitialDaysDFList[i - 7:i])
                mb7 = round(dflb7['price'].mean(), ndp)
                sdb7 = round(dflb7['price'].std(ddof=1), ndp)


                if(math.isnan(m)):
                    m = 0
                if(math.isnan(mb30)):
                    mb30 = 0
                if(math.isnan(mb1)):
                    mb1 = 0
                if(math.isnan(mb7)):
                    mb7 = 0

                if(math.isnan(sd)):
                    sd = 0
                if(math.isnan(sdb30) or sdb30==0):
                    sdb30 = 1
                if(math.isnan(sdb1) or sdb1==0):
                    sdb1 = 1
                if(math.isnan(sdb7) or sdb7==0):
                    sdb7 = 1

                nummdev30 = ((m - mb30) / sdb30)
                nummdev7 = ((m - mb7) / sdb7)
                nummdev1 = ((m - mb1) / sdb1)

                numsddev30 = ((sd - sdb30) / sdb30)
                numsddev7 = ((sd - sdb7) / sdb7)
                numsddev1 = ((sd - sdb1) / sdb1)


                numrows = dfls.shape[0]

                dfls['numdev_means_30'] = [nummdev30] * (numrows)
                dfls['numdev_means_7'] = [nummdev7] * (numrows)
                dfls['numdev_means_1'] = [nummdev1] * (numrows)

                dfls['numdev_deviations_30'] = [numsddev30] * (numrows)
                dfls['numdev_deviations_7'] = [numsddev7] * (numrows)
                dfls['numdev_deviations_1'] = [numsddev1] * (numrows)

                # if(hr == 10 and mn == 0):
                #     print(dfls)
                #     print(dfl)
                #
                #     sys.exit()

                MinutesDFList.append(dfls)

        # print(pd.concat(MinutesDFList).head(50))
        # sys.exit()

        FinalDaysDFList.append(pd.concat(MinutesDFList))


    print(FinalDaysDFList[0])
    stockstatsdf = pd.concat(FinalDaysDFList)
    stockstatsdf.to_pickle(StockstatsDataframePath)
    sys.exit()

# stockstatsdf = pd.read_pickle(StockstatsDataframePath)

stockstatsdf = pd.read_pickle(StockstatsDataframePath)[['Trans.ID','price','binarylabels','numdev_means_30', 'numdev_means_7', 'numdev_means_1', 'numdev_deviations_30', 'numdev_deviations_7', 'numdev_deviations_1']]
# stockstatsdf = pd.read_pickle(StockstatsDataframePath)[['numdev_means_30', 'numdev_means_7', 'numdev_means_1', 'numdev_deviations_30', 'numdev_deviations_7', 'numdev_deviations_1']]
# Keep Trans.ID, binarylabels for visualization
X = stockstatsdf.values
# X = stockstatsdf.values[0:100]
# print('X.shape',X.shape)

numrecords = len(X)

clf = LocalOutlierFactor(n_neighbors=numneighbours, contamination=outliers_fraction)

y_pred = clf.fit_predict(X[:,3:])
scores_pred = clf.negative_outlier_factor_
# The lower, the more normal. Inliers tend to have a LOF score close to 1, while outliers tend to have a larger LOF score.

print('y_pred', y_pred)
print(len(y_pred[y_pred == -1]))
print(len(y_pred))

print('scores_pred', scores_pred)
print(min(scores_pred))
print(max(scores_pred))

print(X[:,2])
print(X[:,1])

print(np.column_stack((X[:,0],X[:,1],X[:,2],y_pred,scores_pred,)))

# TO DO :
# To add first two pcs of the features
# To show visualization for top 100 points

# numtrainrecords = int(math.ceil(0.7 * numrecords))
# numtestrecords = int(math.ceil(0.3 * numrecords))
#
# # clf = LocalOutlierFactor(n_neighbors=numneighbours)
# clf = LocalOutlierFactor(n_neighbors=10)
#
# clf.fit(X[:numtrainrecords])
# # clf.fit(X)
#
# # y_pred = clf.fit_predict(X)
# y_pred = clf._predict(X[-numtestrecords:])
# # y_pred = clf._predict(X)
# # Returns -1 for anomalies/outliers and 1 for inliers.
# print(y_pred)
# print(len(y_pred[y_pred == -1]))
#
# # pred_lof_scores = clf.negative_outlier_factor_
# pred_lof_scores = clf._decision_function(X[-numtestrecords:])
# # pred_lof_scores = clf._decision_function(X)
#
# # Returns The opposite of the Local Outlier Factor of each input samples. The lower, the more abnormal.
# print(pred_lof_scores)
# print(len(pred_lof_scores))



# Uncomment following code for LOF scores on Yahoo data
# tw = 100
# bw = 500
# lofw1 = 30
# lofw2 = 7
# Time windows to discretize daily closing prices

# df = pd.read_csv("/data/achivuku/Documents/financedataanalysis/pricesvolumes.csv")
# cols = [1,2,3,4,6,8,10,12,14,16,18,20,21,22,23,24,26,28,30,32,33,34,36,38,40,42]
# df.drop(df.columns[cols],axis=1,inplace=True)

# stockname = 'ORCL_prices'
# dfstockname = stockname.rstrip('_prices')
#
# stock = df[[stockname]].values
# # bw = len(stock) - tw
# flatstock = [round(x,ndp) for x in list(chain.from_iterable(stock))]
# stockdiff = [0] * (len(stock))
# stocklocalstats = []
# stockmeans = [0] * (len(stock))
# stockdeviations = [0] * (len(stock))
# stockthresholds = [0] * (len(stock))
#
# lofw = lofw1
# lofindex1 = getlofindex(stock, lofw)
# lofw = lofw2
# lofindex2 = getlofindex(stock, lofw)
#
# for i in xrange(len(stock)-1, tw, -tw): # for many benchmarking window, replace all nan with 0
# # for i in xrange(len(stock)-1, len(stock)-bw, -tw): # for one benchmarking window
#     for j in xrange(i, i-tw, -1):
#         stockdiff[j] = round((stock[j] - stock[i-tw-1])[0], ndp)
#     # stockstats.append(stockdiff[i])
#     stocklocalstats = stockdiff[i - tw + 1:i + 1]
#     m = np.mean(np.abs(stocklocalstats))
#     sd = np.std(np.abs(stocklocalstats))
#     t = round(m + nstd * sd, ndp)
#     for j in xrange(i, i - tw, -1):
#         stockmeans[j] = round(m, ndp)
#         stockdeviations[j] = round(sd, ndp)
#         stockthresholds[j] = round(t, ndp)
#
# # m = np.mean(np.abs(stockstats))
# # sd = np.std(np.abs(stockstats))
# # t = m + nstd * sd
#
# # stockdiff = stockdiff[-bw:]
# # m = np.mean(np.abs(stockdiff))
# # sd = np.std(np.abs(stockdiff))
# # t = m + nstd * sd
#
# # print(stockdiff)
# #
# # print(t)
# # print(m)
# # print(sd)
# # print('\n')
#
#
#
# stockstatsdf = pd.DataFrame(
#     {
#         # dfstockname + '_prices': flatstock,
#         # dfstockname + '_pricediffs': stockdiff,
#         dfstockname + '_pricemeans': stockmeans,
#         dfstockname + '_pricedeviations': stockdeviations,
#         dfstockname + '_pricethresholds': stockthresholds,
#         dfstockname + '_pricetimeindex_months': lofindex1,
#         dfstockname + '_pricetimeindex_weeks': lofindex2
#     }
# )
#
# print(stockstatsdf.tail(10))
# # print(stockstatsdf[[dfstockname + '_pricediffs']])
# # print(stockstatsdf[[dfstockname + '_pricemeans']])
# # print(stockstatsdf[[dfstockname + '_pricedeviations']])
# # print(stockstatsdf[[dfstockname + '_pricethresholds']])
# # print(stockstatsdf[[dfstockname + '_pricetimeindex_months']])
# # print(stockstatsdf[[dfstockname + '_pricetimeindex_weeks']])
#
# X = stockstatsdf.values
#
# print(X.shape)
# numrecords = len(X)
# numtrainrecords = int(math.ceil(0.7 * numrecords))
# numtestrecords = int(math.ceil(0.3 * numrecords))
#
# clf = LocalOutlierFactor(n_neighbors=numneighbours)
# clf.fit(X[:numtrainrecords])
#
# y_pred = clf._predict(X[-numtestrecords:])
# # y_pred = clf.fit_predict(X[:numtrainrecords])
# # Returns -1 for anomalies/outliers and 1 for inliers.
# print(y_pred)
# print(len(y_pred[y_pred == -1]))
#
# pred_lof_scores = clf._decision_function(X[-numtestrecords:])
# # Returns The opposite of the Local Outlier Factor of each input samples. The lower, the more abnormal.
# print(pred_lof_scores)
# print(len(pred_lof_scores))
# # print(clf._decision_function(X[:numtrainrecords]))
# # print(clf.negative_outlier_factor_)