# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:43:45 2017

@author: Marijuana
"""

import QM_GC as QG
import numpy as np
import scipy.io as sio
import pickle
from scipy.optimize import minimize

matfile = u'C:\\Users\\Marijuana\\Desktop\\testData.mat'#load .mat file
#data = sio.loadmat(matfile)
#regressors = data['regressors']
#xdep = data['xdep']
#import Optimization as opt
#opt.Midlevelprocess(regressors,xdep[:,1:2],5,0.8,5)
data = sio.loadmat(matfile)
X = data['X']
ret = QG.cca_granger_regress(X,2,1)
#QT = QG.cca_cdff(1.9446984,2,8)

#pkl_file = open('C:\Users\Marijuana\Documents\data.pk1', 'rb')
#data1 = pickle.load(pkl_file)
#betaInitialisation = data1['betaInitialisation']
#trainRegressors = data1['trainRegressors']
#trainxdep = data1['trainxdep']
#alpha = data1['alpha']
#lamda = data1['lamda']
#spikeIndex = data1['spikeIndex']
#nonSpikeIndex = data1['nonSpikeIndex']
#dea = opt.R_QM_REG_der2(betaInitialisation,trainRegressors,trainxdep,alpha,lamda,spikeIndex,nonSpikeIndex)
