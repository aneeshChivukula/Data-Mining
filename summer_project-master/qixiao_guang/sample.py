# -*- coding:utf-8 -*-
import numpy as np

def sampling(X):
    delIndex=[]
    for i,row in enumerate(X):
        if i%10==0:
            continue
        else:
            if row[-1]<0:
                for j in range(i,i-i%10-1,-1):
                    if X[j][-1]==1:
                        delIndex.append(i)
                        break

    new_X1=np.delete(X,delIndex,0)
    return new_X1


