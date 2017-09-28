import numpy as np

def covariance(X,Y,n):
    x_bar = np.mean(X,axis=0)
    y_bar = np.mean(X,axis=0)
    cov_value=0;
    for i in range(1,n+1):
        cov_value = cov_value+X[i]*Y[i]
    return cov_value/n - x_bar*y_bar