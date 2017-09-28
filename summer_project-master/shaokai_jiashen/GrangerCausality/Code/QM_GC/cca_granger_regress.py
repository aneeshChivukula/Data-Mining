import numpy as np


def cca_granger_regress(X, nlags, STATFLAG):
    # -----------------------------------------------------------------------
    #   FUNCTION: cca_granger_regress.py
    #   PURPOSE:  perform multivariate regression with granger causality statistics
    #
    #   INPUT:  X           -   nvar (rows) by nobs (cols) observation matrix
    #           nlags       -   number of lags to include in model
    #           STATFLAG    -   if 1 do F-stats
    #
    #   OUTPUT: ret.covu    -   covariance of residuals for unrestricted model
    #           ret.covr    -   covariance of residuals for restricted models
    #           ret.prb     -   Granger causality probabilities (column causes
    #                           row. NaN for non-calculated entries)
    #           ret.fs      -   F-statistics for above
    #           ret.gc      -   log ratio causality magnitude
    #           ret.doi     -   difference-of-influence (based on ret.gc)
    #           ret.rss     -   residual sum-square
    #           ret.rss_adj -   adjusted residual sum-square
    #           ret.waut    -   autocorrelations in residuals (by Durbin
    #           Watson)
    #           ret.cons    -   consistency check (see cca_consistency.m)

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

    if nvar > nobs:
        raise Exception('error in cca_granger_regress: nvar>nobs, check input matrix')

    # remove sample means if present (no constant terms in this regression)
    m = np.mean(X.T, axis=0)
    if abs(sum(m)) > 0.0001:
        mall = np.tile(m.reshape(m.shape[0], 1), [1, nobs])
        X = X - mall

    # construct lag matrices
    lags = -999 * np.ones((nvar, nobs - nlags, nlags))
    for jj in range(1, nvar + 1):
        for ii in range(1, nlags + 1):
            lags[jj, :, nlags - ii + 1] = X[jj, ii:nobs - nlags + ii - 1]

            # unrestricted regression (no constant term)
    regressors = np.zeros((nobs - nlags, nvar * nlags))
    for ii in range(1, nvar + 1):
        s1 = (ii - 1) * nlags + 1
        regressors[:, s1:s1 + nlags - 1] = np.squeeze(lags[ii, :, :])

    # xdep=X[:,nlags+1:end]';
    # beta=regressors\xdep;
    for ii in range(1, nvar + 1):
        xvec = X[ii, :].reshape(X[ii, :].shape[0], 1)
        xdep = xvec[nlags + 1:xvec.shape[0]]
        beta[:, ii] = np.linalg.inv(regressors) * xdep
        xpred[:, ii] = regressors * beta[:, ii]
        u[:, ii] = xdep - xpred[:, ii]
        RSS1[ii] = np.sum(i * i for i in u[:, ii])
        C[ii] = covariance(u[:, ii], u[:, ii], nobs - nlags)
    covu = cov(u)

    #   A rectangular matrix A is rank deficient if it does not have linearly independent columns.
    #   If A is rank deficient, the least squares solution to AX = B is not unique.
    #   The backslash operator, A\B, issues a warning if A is rank deficient and
    #   produces a least squares solution that has at most rank(A) nonzeros.

    #   restricted regressions (no constant terms)
    for ii in range(1, nvar + 1):
        xvec = X[ii, :].reshape(X[ii, :].shape[0], 1)
        xdep = xvec[nlags + 1:xvec.shape[0]]
        caus_inx = np.setdiff1d(range(1, nvar + 1), ii)
        u_r = np.zeros((nobs - nlags, nvar), float)