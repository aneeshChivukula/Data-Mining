# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:03:50 2017

@author: ShaokaiZhao
"""
# package import
import numpy as np
import Optimization as opt

def covariance(X,Y,n):
    x_bar = np.mean(X,axis=0)
    y_bar = np.mean(Y,axis=0)
    cov_value=0;
    for i in range(0,n):
        cov_value = cov_value+X[i]*Y[i]
    return cov_value/n-x_bar*y_bar

def cca_consistency(X,xpred):
#-----------------------------------------------------------------------
# FUNCTION: cca_consistency.m
# PURPOSE:  check what portion of the correlation structure in data is
#           accounted for by an MVAR model, based on Ding et al 2000
#
# INPUTS:   X         nvar (rows) by nobs (cols) observation matrix
#           xpred     predicted values from the MVAR model
#
# OUTPUT:   cons:  consistency check (near 100% is good)
#
#           Ding, Bressler, Yang, Liang (2000) Biol Cyb 83:35-45
#
#           Seth, A.K. (in preparation), A MATLAB Toolbox for Granger
#           causal connectivity analysis
#-----------------------------------------------------------------------   
    nvar = X.shape[0]
    Rr = np.cov(X)
    Rs = np.cov(xpred.T)
    Rr = Rr.reshape((1,nvar*nvar))
    Rs = Rs.reshape((1,nvar*nvar))
    cons = np.dot(np.abs(Rs-Rr),np.linalg.pinv(np.abs(Rr)))
    cons = (1-cons)*100
    return cons
    
def cca_cdff(x,v1,v2):
#   CDF computes the cumulative 'F' distribution

    p = 0
    t = (v1 <= 0 or v2 <= 0 or np.isnan(x) or np.isnan(v1) or np.isnan(v2))
    s = (x==np.inf) & (not t)
    if np.any(s):
        p = 1
        t = t | s
        
#   Compute P when X > 0
    import scipy.special as ss
    k = np.where(x>0 and (not t) and np.isfinite(v1) and np.isfinite(v2))
    if k[0].shape[0] <> 0:
        xx = x/(x+v2/v1)
        p = ss.betainc(v1/2,v2/2,xx)
    
    from scipy.stats import chi2
    if np.any((not np.isfinite(v1)) or (not np.isfinite(v2))):
        k = np.where(x>0 and (not t) and np.isfinite(v1) and (not np.isfinite(v2)) and v2>0)
        if k[0].shape[0] <> 0:
            p = chi2.cdf(v1*x,v1)
        k = np.where(x>0 and (not t) and (not np.isfinite(v1)) and v1>0 and np.isfinite(v2))
        if k[0].shape[0] <> 0:
            p = 1 - chi2.cdf(v2/x,v2)
        k = np.where(x>0 and (not t) and (not np.isfinite(v1)) and (not np.isfinite(v2)) and v2>0)
        if k[0].shape[0] <> 0:
            p = (x>=1)
    return p
            
def cca_granger_regress(X,nlags,STATFLAG):
# -----------------------------------------------------------------------
#   FUNCTION: cca_granger_regress.py
#   PURPOSE:  perform multivariate regression with granger causality statistics
#
#   INPUT:  X           -   nvar (rows) by nobs (cols) observation matrix
#           nlags       -   number of lags to include in model
#           STATFLAG    -   if 1 do F-stats
#
#   OUTPUT: ret["covu"]    -   covariance of residuals for unrestricted model
#           ret["covr"]    -   covariance of residuals for restricted models
#           ret["prb"]     -   Granger causality probabilities (column causes
#                           row. NaN for non-calculated entries)
#           ret["fs"]      -   F-statistics for above
#           ret["gc"]      -   log ratio causality magnitude
#           ret["doi"]     -   difference-of-influence (based on ret.gc)
#           ret["rss"]     -   residual sum-square
#           ret["rss_adj"] -   adjusted residual sum-square
#           ret["waut"]    -   autocorrelations in residuals (by Durbin
#           Watson)
#           ret["cons"]    -   consistency check (see cca_consistency.m)

#   Written by Anil K Seth Sep 13 2004
#   Updated AKS December 2005
#   Updated AKS November 2006
#   Updated AKS December 2007 to do ratio based causality
#   Updated AKS May 2008, fix nlags = 1 bug.
#   Updated AKS Apr 2009, difference-of-influence and optional stats
#   Updated AKS Aug 2009, specify regressor matrix size in advance
#   Updated AKS Aug 2009, implement whiteness + consistency checks
#   Updated SKZ Jul 2017, transform code from matlab to python
#   Ref: Seth, A.K. (2005) Network: Comp. Neural. Sys. 16(1):35-55
# COPYRIGHT NOTICE AT BOTTOM
# -----------------------------------------------------------------------

# SEE COPYRIGHT/LICENSE NOTICE AT BOTTOM

# figure regression parameters
    size = X.shape
    nobs = size[1]
    nvar = size[0]

    if nvar>nobs :
        raise Exception('error in cca_granger_regress: nvar>nobs, check input matrix')
        
# remove sample means if present (no constant terms in this regression)
    m = np.mean(X.T, axis = 0);
    if abs(sum(m)) > 0.0001:
        mall = np.tile(m.reshape(m.shape[0],1),[1,nobs])
        X = X-mall

# construct lag matrices
    lags = -999*np.ones((nvar,nobs-nlags,nlags))
    for jj in range(0,nvar):
        for ii in range(0,nlags):
            lags[jj,:,nlags-ii-1] = X[jj,ii:nobs-nlags+ii]

# unrestricted regression (no constant term)
    regressors = np.zeros((nobs-nlags,nvar*nlags))
    for ii in range(0,nvar):
        s1 = (ii+1)*nlags-2
        regressors[:,s1:s1+nlags] = np.squeeze(lags[ii,:,:])
        
    beta = np.zeros((nvar*nlags,nvar))
#-------------------raw version-----------------------------
#    xpred = np.zeros((nobs-nlags,nvar))
#    u = np.zeros((nobs-nlags,nvar))   
#-------------------our version-----------------------------
    trteRate = 0.8
    sp = int((nobs-nlags)*trteRate)
    xpred = np.zeros((len(range(sp,(nobs-nlags))),nvar))
    u = np.zeros((len(range(sp,(nobs-nlags))),nvar))  
#-------------------version end-----------------------------

    RSS1 = np.zeros((1,5))
    C = np.zeros((1,5))
    for ii in range(0,nvar):
        xvec = X[ii,:].reshape(X[ii,:].shape[0],1)
        xdep = xvec[nlags:xvec.shape[0]]
#-------------------raw version-----------------------------
#        beta[:,ii:ii+1] = np.dot(np.dot(np.linalg.inv(np.dot(regressors.T,regressors)),regressors.T),xdep)
#        xpred[:,ii:ii+1] = np.dot(regressors,beta[:,ii:ii+1])
#        u[:,ii:ii+1] = xdep-xpred[:,ii:ii+1]
#-------------------our version-----------------------------
        win_length = 5
        if win_length>nobs :
            raise Exception('error in cca_granger_regress: window_length>nobs, check win_length')
        optResult = opt.Midlevelprocess(regressors,xdep,win_length,trteRate,nvar)
        beta[:,ii:ii+1] = optResult['beta']
        testRegressors = optResult['testRegressors']
        testxdep = optResult['testxdep']
        xpred[:,ii:ii+1] = np.dot(testRegressors,beta[:,ii:ii+1])
        u[:,ii:ii+1] = testxdep-xpred[:,ii:ii+1]
#-------------------version end-----------------------------
        RSS1[0,ii] = np.sum(u[:,ii]*u[:,ii])
        C[0,ii] = covariance(u[:,ii],u[:,ii],len(range(sp,(nobs-nlags))))#raw version is nobs-nlags
    covu = np.cov(u.T)   

#   A rectangular matrix A is rank deficient if it does not have linearly independent columns.
#   If A is rank deficient, the least squares solution to AX = B is not unique.
#   The backslash operator, A\B, issues a warning if A is rank deficient and
#   produces a least squares solution that has at most rank(A) nonzeros.

#   restricted regressions (no constant terms)
    RSS0=np.zeros((nvar,nvar))
    S=np.zeros((nvar,nvar))
    covr=np.zeros((nvar,nvar,nvar))
#------------------------our version------------------------
    beta = np.zeros(((nvar-1)*nlags,nvar))
#-----------------------------------------------------------
    for ii in range(0,nvar):
        xvec = X[ii,:].reshape(X[ii,:].shape[0],1)
        xdep = xvec[nlags:xvec.shape[0]]
        caus_inx = np.setdiff1d(range(0,nvar),ii)
#-------------------raw version-----------------------------           
#        u_r = np.zeros((nobs-nlags,nvar),float)
#-------------------our version-----------------------------
        u_r = np.zeros((len(range(sp,(nobs-nlags))),nvar),float)
#-----------------------------------------------------------        
        for jj in range(0,len(caus_inx)):
            eq_inx = np.setdiff1d(range(0,nvar),caus_inx[jj])
            regressors = np.zeros((nobs-nlags,len(eq_inx)*nlags))
            for kk in range(0,len(eq_inx)):
                s1 = kk*nlags
                regressors[:,s1:s1+nlags] = np.squeeze(lags[eq_inx[kk],:,:])
#-------------------raw version-----------------------------            
#            beta_r = np.dot(np.dot(np.linalg.inv(np.dot(regressors.T,regressors)),regressors.T),xdep)
#            temp_r = xdep-np.dot(regressors,beta_r)
#-------------------our version-----------------------------
            win_length = 5
            if win_length>nobs :
                raise Exception('error in cca_granger_regress: window_length>nobs, check win_length')
            optResult = opt.Midlevelprocess(regressors,xdep,win_length,trteRate,nvar-1)
            beta[:,ii:ii+1] = optResult['beta']
            testRegressors = optResult['testRegressors']
            testxdep = optResult['testxdep']   
            temp_r = testxdep-np.dot(testRegressors,beta[:,ii:ii+1])
#-------------------version end-----------------------------            
            RSS0[ii,caus_inx[jj]] = np.sum(temp_r*temp_r)
            S[ii,caus_inx[jj]] = covariance(temp_r,temp_r,len(range(sp,(nobs-nlags))))#raw version is nobs-nlags
            u_r[:,caus_inx[jj]:caus_inx[jj]+1] = temp_r
        covr[:,:,ii] = np.cov(u_r.T)
    
# calc Granger values
    gc = np.ones((nvar,nvar))
    doi = np.ones((nvar,nvar))
# do Granger f-tests if required
    if STATFLAG == 1:
        prb = np.ones((nvar,nvar))
        ftest = np.zeros((nvar,nvar))
        n2 = (nobs-nlags)-(nvar*nlags)
        for ii in range(0,nvar-1):
            for jj in range(ii+1,nvar):
                ftest[ii,jj] = ((RSS0[ii,jj]-RSS1[0,ii])/nlags)/(RSS1[0,ii]/n2)
                prb[ii,jj] = 1 - cca_cdff(ftest[ii,jj],nlags,n2)
                ftest[jj,ii] = ((RSS0[jj,ii]-RSS1[0,jj])/nlags)/(RSS1[0,jj]/n2)
                prb[jj,ii] = 1 - cca_cdff(ftest[jj,ii],nlags,n2)
                gc[ii,jj] = np.log(S[ii,jj]/C[0,ii])
                gc[jj,ii] = np.log(S[jj,ii]/C[0,jj])
    else:
        ftest = -1
        prb = -1
        for ii in range(0,nvar):
            for jj in range(ii+1,nvar):
                gc[ii,jj] = np.log(S[ii,jj]/C(0,ii))
                gc[jj,ii] = np.log(S[jj,ii]/C(0,jj))
                doi[ii,jj] = gc[ii,jj] - gc[jj,ii]
                doi[jj,ii] = gc[jj,ii] - gc[ii,jj]
                
# do r-squared and check whiteness, consistency
    rss = np.zeros((1,nvar))
    waut = np.zeros((1,nvar))
    rss_adj = np.zeros((1,nvar))
    if STATFLAG == 1:
        df_error = (nobs-nlags)-(nvar*nlags)
        df_total = (nobs-nlags)
        for ii in range(0,nvar):
            xvec = X[ii,nlags:X.shape[1]]
            rss2 = np.dot(xvec.T,xvec)
            rss[0,ii] = 1-(RSS1[0,ii]/rss2)
            rss_adj[0,ii] = 1 - ((RSS1[0,ii]/df_error)/(rss2/df_total))
            #waut[ii] = cca_whiteness[X,u[:,ii]]
            waut[0,ii] = -1 #TEMP APR 19 COMPILER ERROR
        cons = cca_consistency(X,xpred)
    else:
        rss = -1
        rss_adj = -1
        waut = -1
        cons = -1

# organize output structure
    ret={"gc":gc,"fs":ftest,"prb":prb,"covu":covu,"covr":covr,"rss":rss,"rss_adj":rss_adj,"waut":waut,"cons":cons,"doi":doi,"type":"td_normal"}
    return ret