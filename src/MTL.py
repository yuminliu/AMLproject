# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:39:16 2017

@author: liuyuming
"""


import numpy as np
#import sys
import updateParameters

########################### main function #####################################
def MTL(X, Y, K, Lambda = 0.1, Mu = 5, Gamma = 0.1, Beta = 5, maxIter=100):
    import time
    #startTime = time.time()
    #X: list, each component is an array for a task
    #Y: list, each component is an array for a task
    #Lambda = 0.1 # regularization coefficient for tr(S'OmegaS)
    #Mu = 5 # regularization coefficient for norm(S)
    #Gamma = 0.1 # regularization coefficient for norm(L)
    #Beta = 5#0.28 # regularization coefficient for norm(Omega)
    #K = 3 # when K=D and gamma=0, then it is the same with MSSL
    ##K = data['K'][0,0]
    #maxIter = 100 # maximum number of iteration
    
    EPS = 1e-3 # tolerance
    T = len(X) # total number of task
    dummyN, D = X[0].shape # D is the dimension of the data
      
    #### initialization
    #W = np.zeros((D,T))
    #for t in range(T):
    #    temp = X[:,:,t].T
    #    b = np.dot(temp,Y[:,t])
    #    A = np.dot(temp,X[:,:,t])
    #    W[:,t] = np.linalg.solve(A,b)
    np.random.seed(1)
    W = np.random.rand(D,T)
    U, s, V = np.linalg.svd(W)
    L = U[:,0:K]
    #L = np.eye(D,K)# D=K
    ## initialize S=inv(L'L)l'W
    A = np.dot(L.T,L)
    b = np.dot(L.T,W)
    S = np.linalg.solve(A,b)
    Omega = np.identity(T) # precision matrix initialization
    
    #### alternatively update parameter matrices
    for nIter in np.arange(0,maxIter):
        print('iter={}'.format(nIter))
    #    print str(nIter) + "th iteration..."
    #    sstartt = time.time()
        ## update S
        #S = updateParameters.update_S_vectorize(X,Y,Omega,L,Lambda,Mu,maxIter=100)
        S = updateParameters.update_S_l1ls(X,Y,Omega,L,Lambda,Mu)
        #S = updateParameters.update_S_minConf(X,Y,Omega,L,Lambda,Mu,max_Iter=500)
        #S = updateParameters.update_S_cvxpy(X,Y,Omega,L,Lambda,Mu)
    #    sendt = time.time()
    #    print str(nIter) + "th S update time is " + str((sendt-sstartt)/60.0) + " minutes" 
    #    ## update L
        L = updateParameters.update_L_closeform(X,Y,S,Gamma)
    #    lendt = time.time()
    #    print str(nIter) + "th L update time is " + str((lendt-sendt)/60.0) + " minutes"  
    
        ## update Omega 
        Omega = updateParameters.update_Omega_admm(S,Lambda,Beta,max_Iter=100)
        #Omega = updateParameters.update_Omega_sklearn(S,Lambda,Beta,max_Iter=500)
    #    oendt = time.time()
    #    print str(nIter) + "th Omega update time is " + str((oendt-lendt)/60.0) + " minutes" 
        #print str(nIter) + "th iteration finished!"
        W_old = W
        W = np.dot(L,S)
        if(np.linalg.norm(W-W_old)<EPS):
            print("W converge!")
            break
    
#    endTime = time.time()
#    print "total runing time is " + str((endTime-startTime)/60.0) + " minutes"  
                
    return W, L, S, Omega           

            
            
            

    