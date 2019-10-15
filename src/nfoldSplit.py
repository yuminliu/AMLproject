# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 15:09:12 2017

@author: liuyuming
"""

import numpy as np


#nthfold, nfold = 1, 5   
##data = np.load('../data/3tasks-nonoverlap-DATA.npy').item()
#data = np.load('../data/schoolData.npy').item()
#X = data['Xtrain']
#Y = data['Ytrain']

def nfoldsplit(X, Y, nfold, nthfold):
    T = len(Y)
    X_train = []
    Y_train = []
    X_cv = []
    Y_cv = []
    for t in range(T):
        N, dummy = Y[t].shape
        #Ncv = int(round(N*cvRatio))
        #Ntrain = N - Ncv
        num = int(np.round(1.0*N/nfold))
        ind = np.arange(N)
        cvstart, cvend = (nthfold-1)*num, nthfold*num
        if(nthfold==nfold):
            cvend = N
            
        indcv = np.arange(cvstart, cvend)### index for cv data
        indtrain = np.setdiff1d(ind, indcv)### index for training data
        X_train.append(X[t][indtrain,:])
        Y_train.append(Y[t][indtrain,:])
        X_cv.append(X[t][indcv,:])
        Y_cv.append(Y[t][indcv,:])
        
    return X_train, X_cv, Y_train, Y_cv
    
    
