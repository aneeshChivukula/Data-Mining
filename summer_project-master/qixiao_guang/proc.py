# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

import sample

#timeIndex
def proctime(c):
    timeindex10=np.zeros(len(c),dtype=int)
    timeindex100=timeindex10
    for i in range(len(c)):
        timeindex10[i]=i%10+1
        timeindex100[i]=i%100+1
    return timeindex10,timeindex100

#Trading price
def cal_mean(c):
    mean_10days=np.zeros(len(c))
    n=len(c)/10
    m=len(c)%10
    s=c[-m:]
    mean_10days[-m:]=np.mean(s).round(6)
    for i in range(n):
        s=c[i*10:(i+1)*10]
        mean_10days[i*10:(i+1)*10]=np.mean(s).round(6)
    return mean_10days

def cal_std(c):
    std_10days=np.zeros(len(c))
    n=len(c)/10
    m=len(c)%10
    s=c[-m:]
    std_10days[-m:]=np.std(s).round(6)
    for i in range(n):
        s=c[i*10:(i+1)*10]
        std_10days[i*10:(i+1)*10]=np.std(s).round(6)
    return std_10days

def cal_mean_std_100days(c):
    var_100days=np.zeros(len(c))
    std_100days=np.zeros(len(c))
    n=len(c)/100
    m=len(c)%100
    for i in range(n):
        var_100days[i*100:(i+1)*100]=np.mean(c[i*100:(i+1)*100]).round(6)
        std_100days[i * 100:(i + 1) * 100] = np.std(c[i * 100:(i + 1) * 100]).round(6)
    var_100days[-m:]=np.mean(c[-m:]).round(6)
    std_100days[-m:] = np.std(c[-m:]).round(6)
    return var_100days,std_100days

def cal_label(c,mean10,mean100,std100):
    label=np.array(mean10,dtype=bool)
    for i in range(len(mean10)):
        if (mean10[i]>mean100[i]+5*std100[i]) or (mean10[i]<mean100[i]-5*std100[i]):
            label[i]=True
        else:
            label[i]=False
    return label

#Domian features
def cal_MaximumYield(h,l):
    return ((np.abs(h-l))/l).round(6)

def cal_CloseYield(c):
    returns=np.zeros(len(c))
    returns[1:]=(np.diff(c)/c[:-1]).round(6)
    return returns

def cal_ATR(h,l,c):
    """ATR : Average True Range"""
    n=len(c)/10
    m=len(c)/10
    previousclose=np.zeros(len(c))
    previousclose[1:len(c)]=c[0:len(c)-1]
    truerange=np.maximum(h-l,h-previousclose,previousclose-l)
    atr=np.zeros(len(c))
    for i in range(1,len(c)):
        atr[i]=(len(c)-1)*atr[i-1]+truerange[i]
        atr[i]/=len(c).round(6)
    return atr

def cal_OBV(c,v):
    """OBV : On-Balance Volume"""
    change=np.zeros(len(c))
    change[1:]=np.diff(c)
    signs=np.sign(change)
    return v*signs

filename=raw_input()
h,l,c,v=np.loadtxt(filename,delimiter=',',usecols=(2,3,5,6),unpack=True)

timeindex10,timeindex100=proctime(c)
X1=np.column_stack((c,timeindex10,timeindex100))

mean10=cal_mean(c)
std10=cal_std(c)
mean100,std100=cal_mean_std_100days(c)
X2=np.column_stack((c,mean10,std10,mean100,std100))

label=cal_label(c,mean10,mean100,std100)

MaxYield=cal_MaximumYield(h,l)
CloseYield=cal_CloseYield(c)
ATR=cal_ATR(h,l,c)
OBV=cal_OBV(c,v)
X3=np.column_stack((c,MaxYield,CloseYield,ATR,OBV,label))

clf=LocalOutlierFactor(n_neighbors=20)
y_score1=clf.fit_predict(X1)
y_score2=clf.fit_predict(X2)
y_score3=clf.fit_predict(X3)

X1=np.column_stack((X1,y_score1))
X2=np.column_stack((X2,y_score2))
X3=np.column_stack((X3,y_score3))
dataframe1=pd.DataFrame(X1,columns=['ClosePrice','TimeIndex10','TimeIndex100','y_score'])
dataframe2=pd.DataFrame(X2,columns=['ClosePrice','Mean10','Std10','Mean100','Std100','y_score'])
dataframe3=pd.DataFrame(X3,columns=['ClosePrice','MaxYiekd','CloseYield','ATR','OBV','Label','y_score'])
dataframe1.to_csv('X1.csv')
dataframe2.to_csv('X2.csv')
dataframe3.to_csv('X3.csv')

sampleX1=sample.sampling(X1)
sampleX2=sample.sampling(X2)
sampleX3=sample.sampling(X3)

print len(sampleX1)
print len(sampleX2)
print len(sampleX3)






















