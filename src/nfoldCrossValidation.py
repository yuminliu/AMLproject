# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:39:16 2017

@author: liuyuming
"""

import numpy as np
#from randomSplit import randomsplit
from nfoldSplit import nfoldsplit
from MTL import MTL
from rmse import rmse

def nfoldCrossValidation(Xtrain,Ytrain, K, maxIter,nfold = 5,LambdaRange=None,MuRange=None,GammaRange=None,BetaRange=None):
    #### split the data into training data and cross validation data
    #X, Xcv, Y, Ycv = randomsplit(Xtrain, Ytrain, cvRatio=0.3)
    #T = len(Xtrain)
    minRMSE = float("inf")
    Lambda_opt, Mu_opt, Gamma_opt, Beta_opt = [], [], [], []
    #dummyN, D = Xtrain[0].shape
#    startL, endL, numL = 0, 1, 1#1    
#    LambdaRange = np.linspace(startL,endL,numL)
#    startM, endM, numM = 0, 5, 1#6    
#    MuRange = np.linspace(startM,endM,numM)
#    startG, endG, numG = 0, 1, 1#11    
#    GammaRange = np.linspace(startG,endG,numG)
#    startB, endB, numB = 0, 5, 1#6    
#    BetaRange = np.linspace(startB,endB,numB)
    parameters = {}
    for Lambda in LambdaRange:
        for Mu in MuRange:
            for Gamma in GammaRange:
                for Beta in BetaRange:
                    print('Lambda={},Mu={},Gamma={},Beta={}'.format(Lambda,Mu,Gamma,Beta))
                    err = np.zeros((nfold,1))
                    for i in range(1,nfold+1):
                        print('i={}'.format(i))
                        X, Xcv, Y, Ycv = nfoldsplit(Xtrain, Ytrain, nfold, i)#ith fold data                   
                        W,L,S,Omega = MTL(X, Y, K, Lambda, Mu, Gamma, Beta,maxIter)        
                        ## cross validation testing error
#                        Ycv_est = []
#                        testrmse = np.zeros((1,T))
#                        for t in range(T):
#                            Xtestt = Xcv[t] # Ntest by D matrix
#                            Ntest, dummyD = Xtestt.shape
#                            Wt = W[:,t] # D by 1 column vector
#                            Ycv_est.append(np.dot(Xtestt,Wt).reshape((Ntest,1)))
#                            temp = 1.0/Ntest*np.sum(np.power((Ycv[t]-Ycv_est[t]),2))
#                            testrmse[0,t] = np.sqrt(temp)
                        testrmse = rmse(Xcv,Ycv,W)
                        err[i-1,0] = np.mean(testrmse)
                    meanTestRMSE = np.mean(err)                    
                    if(minRMSE>meanTestRMSE):
                        minRMSE = meanTestRMSE
                        Lambda_opt, Mu_opt, Gamma_opt, Beta_opt = Lambda,Mu,Gamma,Beta
                        parameters['W'] = W
                        parameters['L'] = L
                        parameters['S'] = S
                        parameters['Omega'] = Omega
        
    return Lambda_opt, Mu_opt, Gamma_opt, Beta_opt, parameters
        


          

    