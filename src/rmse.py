# -*- coding: utf-8 -*-
"""
Created on Mon May 08 12:49:19 2017

@author: liuyuming
"""
import numpy as np
def rmse(Xtest,Ytest,W):
    #### testing error
    T = len(Ytest)
    Ytest_est = []
    testrmse = np.zeros((1,T))
    for t in range(T):
        Xtestt = Xtest[t] # Ntest by D matrix
        Wt = W[:,t] # D by 1 column vector
        #Ytest_est.append(np.dot(Xtestt,Wt).reshape((-1,1)))
        #temp = np.mean((Ytest[t]-Ytest_est[t])**2)
        Ytest_est = np.dot(Xtestt,Wt).reshape((-1,1))
        temp = np.mean((Ytest[t]-Ytest_est)**2) 
        testrmse[0,t] = np.sqrt(temp)
    
#    meanTestRMSE = np.mean(testrmse)
#    print 'the mean test RMSE is ' + str(meanTestRMSE) 
    
    return testrmse
