# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 09:53:15 2017

@author: Marijuana
"""
import numpy as np
import pickle
import random
from scipy.optimize import minimize
def der1(beta,x,y,ii):
    return 2*(np.dot(np.dot(beta.T,x.T)-y.T,x[:,ii:ii+1])/x.shape[0])
    
def der2(x,ii,jj):
    x = x.T
    w = np.dot(x,x.T)
    return 2*w[ii,jj]
    
def R_QM_REG_der2(beta,x,y,alpha,lamda,spikeIndex,nonSpikeIndex):
#   This function was used to get the derivative of objective function, all be called as Jacobian
#   All meaning of parameter same with the notion difined in R_QM_REG
    beta = beta.reshape(len(beta),1)
    y = y.reshape(len(y),1)
    Norm = np.mean((np.dot(beta.T,x[nonSpikeIndex,:].T)-y[nonSpikeIndex])**2)
    Shft = np.mean((np.dot(beta.T,x[spikeIndex,:].T)-y[spikeIndex])**2)
    der_Normi = np.zeros((len(beta),))
    der_Shfti = np.zeros((len(beta),))
    der2_Norm = np.zeros((len(beta),len(beta)))
    der2_Shft = np.zeros((len(beta),len(beta)))
    dev2 = np.zeros((len(beta),len(beta)))
    for ii in range(0,len(beta)):
        der_Normj = np.zeros((len(beta),))
        der_Shftj = np.zeros((len(beta),))
        der_Normi[ii] = der1(beta,x[nonSpikeIndex,:],y[nonSpikeIndex],ii)
        der_Shfti[ii] = der1(beta,x[spikeIndex,:],y[spikeIndex],ii)                
        for jj in range(0,len(beta)):
            der_Normj[jj] = der1(beta,x[nonSpikeIndex,:],y[nonSpikeIndex],jj)
            der_Shftj[jj] = der1(beta,x[spikeIndex,:],y[spikeIndex],jj)
            der2_Norm[ii,jj] = der2(x[nonSpikeIndex,:],ii,jj)
            der2_Shft[ii,jj] = der2(x[spikeIndex,:],ii,jj)
            part1 = -0.5*((Norm**2+alpha*Shft**2)/(1+alpha))**-1.5*((2*Norm*der_Normj[jj]+2*alpha*Shft*der_Shftj[jj])/(1+alpha))*((2*Norm*der_Normi[ii]+2*alpha*Shft*der_Shfti[ii])/(1+alpha))
            part2 = ((Norm**2+alpha*Shft**2)/(1+alpha))**-0.5*((2*(der_Normj[jj]*der_Normi[ii]+Norm*der2_Norm[ii,jj])+2*alpha*(der_Shftj[jj]*der_Shfti[ii]+Shft*der2_Shft[ii,jj]))/(1+alpha))
            dev2[ii][jj] = 0.5*(part1+part2)
    dev2 = dev2 + 2*lamda*np.eye(dev2.shape[0])
    return dev2

def R_QM_REG_der(beta,x,y,alpha,lamda,spikeIndex,nonSpikeIndex):
#   This function was used to get the derivative of objective function, all be called as Jacobian
#   All meaning of parameter same with the notion difined in R_QM_REG
    beta = beta.reshape(len(beta),1)
    y = y.reshape(len(y),1)
    Norm = np.mean((np.dot(beta.T,x[nonSpikeIndex,:].T)-y[nonSpikeIndex])**2)
    Shft = np.mean((np.dot(beta.T,x[spikeIndex,:].T)-y[spikeIndex])**2)
    der_Norm = np.zeros((len(beta),))
    der_Shft = np.zeros((len(beta),))
    der = np.zeros((len(beta),))
    for ii in range(0,len(beta)):
        der_Norm[ii] = der1(beta,x[nonSpikeIndex,:],y[nonSpikeIndex],ii)
        der_Shft[ii] = der1(beta,x[spikeIndex,:],y[spikeIndex],ii)
        der[ii] = 0.5*((Norm**2+alpha*Shft**2)/(1+alpha))**0.5*((2*Norm*der_Norm[ii]+2*alpha*Shft*der_Shft[ii])/(1+alpha))+2*lamda*beta[ii]
    return der

def R_QM_REG(beta,x,y,alpha,lamda,spikeIndex,nonSpikeIndex):
#   This function include QM model based liner regression and regularization item
#   beta       -  a column vector which represent W(weight) in general regression model
#   x          -  regression sourse
#   y          -  regression destination
#   alpha      -  weight for the distribution-shift samples
#   lamda      -  parameter of regularization item
#   m1 = len(nonSpikeIndex)
#   m2 = len(spikeIndex)
    beta = beta.reshape(len(beta),1)
    y = y.reshape(len(y),1)
    Norm = np.mean((y[nonSpikeIndex].T - np.dot(x[nonSpikeIndex,:],beta))**2)
    Shft = np.mean((y[spikeIndex].T - np.dot(x[spikeIndex,:],beta))**2)
    return np.sqrt((Norm**2+alpha*Shft**2)/(1+alpha))+lamda*np.dot(beta.T,beta)
    
    
def Midlevelprocess(regressors,xdep,t_win,trteRate,nvar):
#   regressor  -  regression sourse
#   xdep       -  regression destination 
#   t_win      -  length of time window
#   trteRate   -  ratio of train data and test data. e.g. (number of train data)/(number of all data)
#   nvar       -  number of variable
#-------Parameter setting--------------------------
    nlags = regressors.shape[1]/nvar
    Nsample = regressors.shape[0]
    k = 2
    lamda = 1.0/Nsample*trteRate
    constantAlpha = 0
#--------------------------------------------------
    Idx = random.sample(range(0,Nsample), Nsample)
    sp = int(Nsample*trteRate)
    trainIndex = Idx[0:sp]
    testIndex = Idx[sp:len(Idx)]
    trainRegressors = regressors[trainIndex,:]
    testRegressors = regressors[testIndex,:]
    trainxdep = xdep[trainIndex]
    testxdep = xdep[testIndex]
    isSpike = np.zeros((len(trainIndex),))
    xdep = xdep.reshape(len(xdep),1)
    for i in range(0,len(trainIndex)):
        if trainIndex[i]<t_win:
            relatedPoints = xdep[0:trainIndex[i]+1]
        else:
            relatedPoints = xdep[(trainIndex[i]-t_win):(trainIndex[i])]
        tempMean = np.mean(relatedPoints,axis=0)
        tempStd = np.std(relatedPoints,axis=0)
        if(xdep[trainIndex[i]]>tempMean+k*tempStd or xdep[trainIndex[i]]<tempMean-k*tempStd):
            isSpike[i] = 1
    spikeIndex = np.where(isSpike)[0]
    nonSpikeIndex = np.where(1-isSpike)[0]
    
    if constantAlpha == 0:
        alpha = len(spikeIndex)/len(nonSpikeIndex)
    
    betaInitialisation = np.zeros((nvar*nlags,1))
    
    print R_QM_REG(betaInitialisation,trainRegressors,trainxdep,alpha,lamda,spikeIndex,nonSpikeIndex)
#        data1 = {'betaInitialisation':betaInitialisation,'trainRegressors':trainRegressors,'trainxdep':trainxdep,'alpha':alpha,'lamda':lamda,'spikeIndex':spikeIndex,'nonSpikeIndex':nonSpikeIndex}
#        output = open('data.pk1','wb')
#        pickle.dump(data1,output)
#        output.close()
    #res = minimize(R_QM_REG,betaInitialisation,args=(trainRegressors,trainxdep,alpha,lamda,spikeIndex,nonSpikeIndex),method='trust-ncg',jac=R_QM_REG_der,hess=R_QM_REG_der2)
    res = minimize(R_QM_REG,betaInitialisation,args=(trainRegressors,trainxdep,alpha,lamda,spikeIndex,nonSpikeIndex))
#        print res.x
    print R_QM_REG(res.x,trainRegressors,trainxdep,alpha,lamda,spikeIndex,nonSpikeIndex)
#        R_QM_REG(betaInitialisation,trainRegressors,trainxdep,alpha,lamda,spikeIndex,nonSpikeIndex)
    res.x = res.x.reshape(len(res.x),1)
    optResult = {'beta':res.x,'testRegressors':testRegressors,'testxdep':testxdep}
    return optResult
        
        
            
        
    
    