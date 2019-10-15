# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 17:04:29 2017

@author: liuyuming
"""

########################### main function #####################################
import time
startTime = time.time()
import numpy as np
from nfoldCrossValidation import nfoldCrossValidation
from MTL import MTL
from rmse import rmse
import sys
#### setting parameters
K = 4 #50
#### read in data       
datapath = '../data/'
datafilename = 'toy-3tasks-nonoverlap-DATA'
data = np.load(datapath+datafilename+'.npy',allow_pickle=True).item()
Xtrain = data['Xtrain']
Ytrain = data['Ytrain'] 
Xtest = data['Xtest']
Ytest = data['Ytest']
T = len(Xtrain)    
maxIter = 100


#### do cross validation to get optimized parameters   
#LambdaRange = [0.001]#[0.001,0.1]
#MuRange = [0.001]#[0.001,0.1]
#GammaRange = [0.001,0.1,10]
#BetaRange = [0.001,0.1,10]  
# =============================================================================
# Lambda,Mu,Gamma,Beta, Parameters = nfoldCrossValidation(Xtrain,Ytrain, K, maxIter,nfold = 5,
#                                                         LambdaRange=LambdaRange,
#                                                         MuRange=MuRange,
#                                                         GammaRange=GammaRange,
#                                                         BetaRange=BetaRange) 
# #Lambda,Mu,Gamma,Beta = 1e-3, 1e-3, 0.1, 1e-3
# =============================================================================
Lambda,Mu,Gamma,Beta = 0.1, 0.1, 0.1, 50. #toy-3tasks-nonoverlap-DATA
#### train on all training data to get results
W,L,S,Omega = MTL(Xtrain, Ytrain, K, Lambda, Mu, Gamma, Beta, maxIter)
            
#### testing error
testrmse = rmse(Xtest,Ytest,W)
meanTestRMSE = np.mean(testrmse)
print('the mean test RMSE is ' + str(meanTestRMSE))

#### save results
import scipy.io as sio
result = {}
result['W_est'] = W
result['L_est'] = L
result['S_est'] = S
result['Omega'] = Omega
result['testrmse'] = testrmse
result['Lambda'] = Lambda
result['Mu'] = Mu
result['Gamma'] = Gamma
result['Beta'] = Beta

para = 'Lambda'+str(Lambda)+'_Mu'+str(Mu)+'_Gamma'+str(Gamma)+'_Beta'+str(Beta)
filename = '../results/RESULT_' + datafilename+'_'+para+'_' + str(K) + '_' + str(meanTestRMSE) + '.mat'
#sio.savemat(filename, result) 

import matplotlib.pyplot as plt
ax = plt.imshow(np.abs(S))
#plt.gray()
plt.show()


endTime = time.time()
print("total runing time is " + str((endTime-startTime)/60.0) + " minutes")






