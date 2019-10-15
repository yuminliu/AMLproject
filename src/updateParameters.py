# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:46:13 2017

@author: liuyuming
"""

import numpy as np


#### update S using minConf_PQN package
#from autograd import grad
#from minConf_PQN import minConf_PQN
#def update_S_minConf(X,Y,Omega,L,Lambda=1.0,Mu=1.0,max_Iter=500):
#    T = len(Y)
#    D, K = L.shape
#    
#    A = np.zeros((K*T,K*T))
#    b = np.zeros((K*T,1))
#    for t in range(T):
#        Nt = len(Y[t])
#        temp1 = np.dot(X[t],L)
#        A[t*K:(t+1)*K,t*K:(t+1)*K] = 1.0/Nt*np.dot(temp1.T,temp1)
#        b[t*K:(t+1)*K,0] = 1.0/Nt*np.dot(Y[t].T,temp1) 
# 
#    M = np.kron(Omega,np.eye(K))
#    B = A + Lambda*M
#    #print B==B.T
#    
#    def fsv(xx,BB=B,bb=b,Muu=Mu):
#        f = np.dot(np.dot(xx.T,BB),xx)-2*np.dot(bb.T,xx)+Muu*np.linalg.norm(xx,ord=1)
#        
#        grad_fsv = grad(fsv)
#        g = grad_fsv(xx)
#        return (f,g)
#    
#    funObj = lambda x: fsv(x)
#    funProj = lambda v: v
#    Sv_init = np.linalg.solve(B,b)
#    options = {'verbose':2,'optTol':1e-4,'maxIter':10}
#    
#    (Sv, f, funEvals) = minConf_PQN(funObj, Sv_init, funProj, options=options)
#    S = Sv.reshape((T,K)).T 
#    return S
#    




##### update S using l1ls package
#import l1ls as llss
from l1_ls import l1ls
def update_S_l1ls(X,Y,Omega,L,Lambda=1.0,Mu=1.0):
    rel_tol = 1e-4
    T = len(Y)
    D, K = L.shape
    
    A = np.zeros((K*T,K*T))
    b = np.zeros((K*T,1))
    for t in range(T):
        Nt = len(Y[t])
        temp1 = np.dot(X[t],L)
        #temp2 = np.dot(temp1.T,temp1)
        A[t*K:(t+1)*K,t*K:(t+1)*K] = 1.0/Nt*np.dot(temp1.T,temp1)
        #yt = Y[:,[t]]
        b[t*K:(t+1)*K,0] = 1.0/Nt*np.dot(Y[t].T,temp1) 
    
    M = np.kron(Omega,np.eye(K))
    B = A + Lambda*M
    
    
    if(abs(Mu)<=1e-10):
        Sv = np.linalg.solve(B,b)
        S = Sv.reshape((T,K)).T
        return S
    
    
    
    
    
    C = np.linalg.cholesky(B).T
    d = np.linalg.solve(C.T,b)
    
    #[Sv, status, hist] = llss.l1ls(C, d, Mu, tar_gap=rel_tol, quiet=True, eta=1e-3, pcgmaxi=5000)
    [Sv, status, hist] = l1ls(C, d, Mu, tar_gap=rel_tol, quiet=True, eta=1e-3, pcgmaxi=5000)
#    print "status is"
#    print status
    S = Sv.reshape((T,K)).T 
    return S
    





#### update S using ADMM with vectorize Sv
#def softThres(a, k):
#    return np.maximum(0,a-k) - np.maximum(0,-a-k)
#
#def update_S_vectorize(X,Y,Omega,L,Lambda=1.0,Mu=1.0,ro=10.0,maxIter=500):
#    ABSTOL = 1e-5
#    RELTOL = 1e-5
#    #### vectorize matrix S to vector Sv
#    #from scipy.linalg import block_diag
#    T = len(Y)
#    D, K = L.shape
##    A = [[]]
##    b = [[]]
##    for t in range(T):
##        temp1 = np.dot(X[:,:,t],L)
##        temp2 = np.dot(temp1.T,temp1)
##        A = block_diag(A,temp2)
##        
##        yt = Y[:,t].reshape((N,1))
##        temp3 = np.dot(yt.T,temp1)
##        #print temp3.shape
##        b = np.concatenate((b,temp3),axis=1)
##    b = b.T
#    
#    A = np.zeros((K*T,K*T))
#    b = np.zeros((K*T,1))
#    for t in range(T):
#        Nt = len(Y[t])
#        temp1 = np.dot(X[t],L)
#        #temp2 = np.dot(temp1.T,temp1)
#        A[t*K:(t+1)*K,t*K:(t+1)*K] = 1.0/Nt*np.dot(temp1.T,temp1)
#        #yt = Y[:,[t]]
#        b[t*K:(t+1)*K,0] = 1.0/Nt*np.dot(Y[t].T,temp1) 
#    
#
#    #print b.shape    
#    M = np.kron(Omega,np.eye(K))
#    B = A + Lambda*M
#    
#    #print (B == B.T)
#    
#    #Svv = np.linalg.solve(B+B.T,2*b)
#    #### solve Sv using ADMM
#    Sv = np.random.rand(K*T,1)
#    z = np.random.rand(K*T,1)
#    z_old = z
#    u = np.random.rand(K*T,1)
#    #u_old = u
#    #ro = 100.0
#    #Mu = 0
#    #maxIter = 1000
#    for k in range(maxIter):
#        AA = 2*B + ro*np.eye(K*T)
#        bb = ro*(z-u) + 2*b
#        Sv = np.linalg.solve(AA,bb)
#        
#        z_old = 1.0*z
#        #z = softThres(Sv+u,Lambda/ro)
#        z = softThres(Sv+u,Mu/ro)
##        #u_old = 1.0*u
##        #u = u_old + Sv - z
#        u = u + Sv - z
#        
#    
#        r_norm = np.linalg.norm(Sv-z)
#        s_norm = np.linalg.norm(-ro*(z-z_old))     
#        eps_pri = np.sqrt(K*T)*ABSTOL + RELTOL*max(np.linalg.norm(Sv),np.linalg.norm(z))
#        eps_dual = np.sqrt(K*T)*ABSTOL + RELTOL*np.linalg.norm(ro*u)
#            
#        if(r_norm<=eps_pri and s_norm<=eps_dual):
#            print "S converged!"
#            break
#
#    S = Sv.reshape((T,K)).T   
#
#    return S
#  
##    ## solve Sv
##    import cvxpy as cvx
##    c = np.linalg.solve(B,b)
##    Sv = cvx.Variable(K*T,1)
##    #loss = Sv.T*B*Sv - 2*b.T*Sv
##    loss = cvx.quad_form( Sv - c, B )
##    regularization = Mu*cvx.sum_entries(cvx.abs(Sv))
##    objective = cvx.Minimize(loss+regularization)
##    problem = cvx.Problem(objective)
##    problem.solve() # use default solver
##    S = Sv.value.reshape((T,K)).T
##    return S
    



#### update Omega using sklearn package
#from sklearn import covariance
#def update_Omega_sklearn(S,Lambda,Beta,max_Iter=500):
#    Tol= 1e-4
#    K, T = S.shape
#    Sbar = (float(Lambda)/K)*np.dot(S.T,S)
#    beta = float(Beta)/K
#    invOmega, Omega = covariance.graph_lasso(emp_cov=Sbar, 
#                                             alpha=beta,
#                                             cov_init=None,
#                                             mode='cd', 
#                                             tol=Tol, 
#                                             enet_tol=0.0001, 
#                                             max_iter=max_Iter, 
#                                             verbose=False, 
#                                             return_costs=False,
#                                             eps=2.2204460492503131e-16,
#                                             return_n_iter=False)
#    
#    return Omega



#### function for update Omega using ADMM
def softhreshol(a, b):
    return np.maximum(0,a-b) - np.maximum(0,-a-b)

#### function for update Omega using ADMM
def update_Omega_admm(S,Lambda,Beta,ro=10.0,max_Iter=500):
    ABSTOL = 1e-5
    RELTOL = 1e-5
    EPS = 1e-8
    K, T = S.shape
    Sbar = (float(Lambda)/K)*np.dot(S.T,S)
    beta = float(Beta)/K
    Z = np.zeros(Sbar.shape)
    U = np.zeros(Sbar.shape)
    Zold = np.zeros(Sbar.shape)
    for oIter in range(max_Iter):
        temp = ro*(Z-U) - Sbar
        d,Q = np.linalg.eig(temp)
        
        d = np.real(d)
        d[np.abs(d)<EPS] = 0 # cast very small values to be zeros
        dd = (d + np.sqrt(d**2+4*ro))/float(2*ro)
        D = np.diag(dd)
        Omega = np.dot(np.dot(Q,D),Q.T)
        Omega = np.real(Omega)
        Zold = 1.0*Z
        Z = softhreshol(Omega+U, beta/ro)      
        U = U + Omega - Z
        
        r_norm = np.linalg.norm(Omega-Z)
        s_norm = np.linalg.norm(-ro*(Z-Zold))
            
        eps_pri = np.sqrt(T*T)*ABSTOL + RELTOL*max(np.linalg.norm(Omega),np.linalg.norm(Z))
        eps_dual = np.sqrt(T*T)*ABSTOL + RELTOL*np.linalg.norm(ro*U)

            
        if(r_norm<=eps_pri and s_norm<=eps_dual):
            print("Omega converged!")
            break

    return Omega




#### update L with close form solution
def update_L_closeform(X,Y,S,gamma):
    dummyN, D = X[0].shape
    K, T = S.shape
    
    A = np.zeros((D*K,1))
    B = np.zeros((D*K,D*K))
    for t in range(T):
        Xt = X[t]
        Yt = Y[t]
        Nt, dummyD = Xt.shape
        At = np.zeros((Nt,1))
        for j in range(K):
            At = np.concatenate((At,Xt*S[j,t]),1)
        At = At[:,1:]
        Bt = np.dot(At.T,At)
        B = B + 1.0/Nt*Bt      
        A = A + 1.0/Nt*np.dot(At.T,Yt)
    
    B = B + gamma*np.eye(D*K)
    Lv = np.linalg.solve(B,A)
    L = Lv.reshape((K,D)).T
    
    return L


##### Obsolete functions
##### update S using cvxpy package
#def update_S_cvxpy(X,Y,Omega,L,Lambda=1.0,Mu=1.0):
#    #### vectorize matrix S to vector Sv
#    #from scipy.linalg import block_diag
#    T = len(Y)
#    D, K = L.shape    
#    A = np.zeros((K*T,K*T))
#    b = np.zeros((K*T,1))
#    for t in range(T):
#        Nt = len(Y[t])
#        temp1 = np.dot(X[t],L)
#        A[t*K:(t+1)*K,t*K:(t+1)*K] = 1.0/Nt*np.dot(temp1.T,temp1)
#        b[t*K:(t+1)*K,0] = 1.0/Nt*np.dot(Y[t].T,temp1) 
#
#    #print b.shape    
#    M = np.kron(Omega,np.eye(K))
#    B = A + Lambda*M
#    #Sv = np.random.rand(K*T,1)
#
#    ## solve Sv
#    import cvxpy as cvx
#    c = np.linalg.solve(B,b)
#    Sv = cvx.Variable(K*T,1)
#    #loss = Sv.T*B*Sv - 2*b.T*Sv
#    loss = cvx.quad_form( Sv - c, B )
#    regularization = Mu*cvx.sum_entries(cvx.abs(Sv))
#    objective = cvx.Minimize(loss+regularization)
#    problem = cvx.Problem(objective)
#    problem.solve() # use default solver
#    S = Sv.value.reshape((T,K)).T
#    return S








#def softhreshol(a, b):
#    return np.maximum(0,a-b) - np.maximum(0,-a-b)
#
#def update_S_admm(X,Y,L,Omega,S,K,Lambda,mu,ro=1,max_Iter = 500):
#    
#    ABSTOL = 1e-5
#    RELTOL = 1e-5
#    
#    N, D, T = X.shape
#    Snew = np.zeros(S.shape) # store the updated St
#    for t in range(T):
#        Xt = X[:,:,t]
#        yt = Y[:,t].reshape(N,1) # N by 1 vector
#        temp1 = np.dot(np.transpose(L),np.transpose(Xt))
#        temp2 = np.dot(temp1,Xt)
#        temp3 = np.dot(temp2,L)
#        P = 2/N*temp3+2*Lambda*Omega[t,t]*np.identity(K)
#        Snoi = np.zeros((K,1))
#        for i in range(0,T):
#            if(i==t):
#                continue
#            temp = S[:,i].reshape(K,1)
#            Snoi = Snoi + Omega[t,i]*(temp)
#
#        temp1 = np.dot(L.T,Xt.T)
#        q = 2*Lambda*Snoi - (2.0/N)*np.dot(temp1,yt)
#        Z = np.zeros((K,1))
#        U = np.zeros((K,1))
#        Zold = np.zeros((K,1))
#        for sIter in range(max_Iter):
#            A = P + ro*np.identity(K)
#            b = ro*(Z-U)-q  
#            St = np.linalg.solve(A,b)
#            Zold = 1*Z
#            Z = softhreshol(St+U, mu/ro)
#            U = U + St - Z
#            
#            r_norm = np.linalg.norm(St-Z)
#            s_norm = np.linalg.norm(-ro*(Z-Zold))
#            
#            eps_pri = np.sqrt(K)*ABSTOL + RELTOL*max(np.linalg.norm(St),np.linalg.norm(Z))
#            eps_dual = np.sqrt(K)*ABSTOL + RELTOL*np.linalg.norm(ro*U)
#            
#            if(r_norm<=eps_pri and s_norm<=eps_dual):
#                print "S"+str(t)+" converged!"
#                break
#        
#        ## concatenate St to be matrix S
#        Snew[:,t] = St.reshape(K)
#    
#    #print Snew-S
#
#    return Snew



### for test only
#beta = 0.28
#ro = 1
#max_Iter = 500
##Lambda = 1
#def update_Omega_admm(Smat,K,Lambda,Beta,ro=1,max_Iter = 500):
#    ABSTOL = 1e-5
#    RELTOL = 1e-5
#    k, T = Smat.shape
#    #Sbar = (float(Lambda)/K)*np.dot(Smat.T,Smat)
#    #Sbar = np.dot(Smat.T,Smat)
#    Sbar = np.cov(Smat.T) ### TODO: figure out which one is better
#    
#    #beta = float(Beta)/K
#    beta = Beta
#    
#    Z = np.zeros(Sbar.shape)
#    U = np.zeros(Sbar.shape)
#    Zold = np.zeros(Sbar.shape)
#    ALFA = 1.4
#    AA = np.zeros(6)
#    for oIter in range(max_Iter):
#        temp = ro*(Z-U) - Sbar
#        d,Q = np.linalg.eigh(temp)
#        
#        #Q = np.real(Q)
#        AA = np.dot(Q,Q.T)
#        d = np.real(d)
#        #d[np.abs(d)<EPS] = 0 # cast very small values to be zeros
#        dd = (d + np.sqrt(d**2+4*ro))/float(2*ro)
#        D = np.diag(dd)
#        Omega = np.dot(np.dot(Q,D),Q.T)
#        #Omega = np.real(Omega)
#        Zold = 1.0*Z
#        
#    #    Z = softhreshol(Omega+U, beta/ro)
#        Omega_hat = ALFA*Omega + (1-ALFA)*Zold;
#        Z = softhreshol(Omega_hat+U, beta/ro)
#              
#        U = U + Omega_hat - Z
#        
#        r_norm = np.linalg.norm(Omega-Z)
#        s_norm = np.linalg.norm(-ro*(Z-Zold))
#            
#        eps_pri = np.sqrt(T*T)*ABSTOL + RELTOL*max(np.linalg.norm(Omega),np.linalg.norm(Z))
#        eps_dual = np.sqrt(T*T)*ABSTOL + RELTOL*np.linalg.norm(ro*U)
#            
#        if(r_norm<=eps_pri and s_norm<=eps_dual):
#            print "Omega converged!"
#            break
#  
##    ## save results
##    import scipy.io
##    Data = {}
##    Data['Omega'] = Omega
##    Data['Sbar'] = Sbar
##    Data['Q'] = Q
##    Data['D'] = D
##    Data['AA'] = AA
##    scipy.io.savemat('C:\\Users\\YM\\Documents\\MATLAB\\andreric-mssl-code\\src\\Omega.mat', Data)
#
#    return Omega



##### function for update L using sklearn package
#from scipy.optimize import minimize
#
#def function_L(L,X,Y,S,K,gamma):
#    N, D, T = X.shape
#    funValue = 0
#    for t in range(T):
#        Xt = X[:,:,t]
#        yt = Y[:,t].reshape(N,1) # N by 1 vector
#        St = S[:,t].reshape(K,1) # K by 1 vector
#        temp = yt - np.dot(np.dot(Xt,L),St)
#        funValue += np.linalg.norm(temp)**2
#        
#    funValue = 1.0/N*funValue + gamma*np.linalg.norm(L,'fro')**2
#    return funValue
#
#def derivativefunc_L(L,X,Y,S,K,gamma):
#    N, D, T = X.shape
#    funJacobian = np.zeros((D,K))
#    for t in range(T):
#        Xt = X[:,:,t]
#        yt = Y[:,t].reshape(N,1) # N by 1 vector
#        St = S[:,t].reshape(K,1) # K by 1 vector
#        temp1 = np.dot(Xt.T,Xt)
#        temp2 = np.dot(St,St.T)
#        temp3 = np.dot(np.dot(temp1,L),temp2)
#        temp4 = np.dot(np.dot(Xt.T,yt),St.T)
#        funJacobian += temp3 - temp4
#        
#    funJacobian = 2.0/N*funJacobian + 2*gamma*L
#    return funJacobian
#
#def update_L_sd(X,Y,L0,S,K,gamma,step=10):
#    maxlIter = 500
#    L = 1*L0
#    Lold = 1*L0
#    Ltemp = 1*L0
#    count = 0
#    for lIter in range(maxlIter):
#        Ltemp = Lold - step*derivativefunc_L(L,X,Y,S,K,gamma)
#        fold = function_L(Lold,X,Y,S,K,gamma)
#        fnew = function_L(Ltemp,X,Y,S,K,gamma)
#        while(fnew>fold):
#            count = 0
#            step = 0.8*step
#            Ltemp = Lold - step*derivativefunc_L(L,X,Y,S,K,gamma)
#            fold = function_L(Lold,X,Y,S,K,gamma)
#            fnew = function_L(Ltemp,X,Y,S,K,gamma)
#   
#        Lold = L
#        L = Ltemp
#        count += 1
#        if(count>=10):
#            count = 0
#            step = 2*step
#        
#        if(np.linalg.norm(L-Lold)<0.01*np.linalg.norm(L)):
#            print "L converge!"
#            break
#        
#    return L
#
#
#
#def update_L_gd(X,Y,L,S,K,gamma):
#    result = minimize(function_L,L,(X,Y,S,K,gamma),method='BFGS',jac=derivativefunc_L)
#    L = result.x
#    return L
#
#import cvxpy as cvx
#def update_L_cvx(X,Y,L,S,K,gamma):
#    N, D, T = X.shape
#    L = cvx.Variable(D,K)
#    loss = 0
#    for t in range(T):
#        Xt = X[:,:,t]
#        yt = Y[:,t].reshape(N,1) # N by 1 vector
#        St = S[:,t].reshape(K,1) # K by 1 vector
#        loss += cvx.norm(yt - Xt*L*St)**2
#    objective = cvx.Minimize(loss + gamma*cvx.norm(L,'fro'))
#    prob = cvx.Problem(objective)
#    prob.solve(solver=cvx.SCS)
#    
#    return L.value
#



#def update_St_cvx(X,Y,L,K,mu,S):
#    import cvxpy as cvx
#    N, D, T = X.shape
#    #L = cvx.Variable(D,K)
#    #S = np.zeros(K,T)
#    #loss = 0
#    for t in range(T):
#        Xt = X[:,:,t]
#        yt = Y[:,t].reshape(N,1) # N by 1 vector
#        St = cvx.Variable(K,1)
#        #St = S[:,t].reshape(K,1) # K by 1 vector
#        temp = np.dot(Xt,L)
#        #loss += cvx.norm(yt - temp*St)**2
#        loss = cvx.norm(yt - temp*St)**2
#        objective = cvx.Minimize(loss + mu*cvx.norm(St,1))
#        prob = cvx.Problem(objective)
#        prob.solve(solver=cvx.SCS)
#        S[:,t] = St.value.reshape(K)
#    
#    return S#L.value


#### wrong expression
#from createPermutationMatrix import createP
#from createSparseAC import createAC
#def update_S_closeform(X,Y,Omega,K,Lambda=1.0):
#    #Seps = 1e-3
#    N, T = Y.shape
#    #N, K, T = X.shape
#    ## TODO: decide dimension D of the pseudo data XL
#    P = createP(K,T)
#    x,y = createAC(X,Y)
#    Svec = np.zeros((K*T,1))
#    L = np.kron(np.eye(K),Omega)
#    Xls = x + Lambda*np.dot(np.dot(P,L),P.T)
#    Svec = np.linalg.solve(Xls,y)
#    Smat = Svec.reshape((T,K)).T 
#### save results
##import scipy.io
##Data = {}
##Data['Smat'] = Smat
##Data['Svec'] = Svec
##Data['P'] = P
##Data['x'] = x
##Data['y'] = y
##scipy.io.savemat('C:\\Users\\YM\\Documents\\MATLAB\\andreric-mssl-code\\src\\Smat.mat', Data)
#    return Smat

### for test only
#from scipy.io import loadmat
#data = loadmat('./data/4tasks-overlap-DATA.mat')
#X = data['Xtrain']
#Y = data['Ytrain']
#N, D, T = X.shape
#Omega = np.identity(T)
#K = D
#Lambda = 1.0

#### update S using cvxpy package
#def update_S_cvx(X,Y,Omega,L,K,mu,Lambda):
#    import cvxpy as cvx
#    N, D, T = X.shape
#    #L = cvx.Variable(D,K)
#    S = cvx.Variable(K,T)
#    loss = 0
#    for t in range(T):
#        Xt = X[:,:,t]
#        yt = Y[:,t].reshape(N,1) # N by 1 vector
#        #St = cvx.Variable(K,1)
#        St = S[:,t] # K by 1 vector?
#        temp = np.dot(Xt,L)
#        loss += cvx.norm(yt - temp*St)**2
#        #loss = cvx.norm(yt - temp*St)**2
#    loss += Lambda*cvx.trace(S*Omega*S.T)
#    regularization = mu*cvx.sum_entries(cvx.abs(S))
#    #objective = cvx.Minimize(loss + mu*cvx.norm(S,1))
#    objective = cvx.Minimize(loss + regularization)
#    prob = cvx.Problem(objective)
#    prob.solve(solver=cvx.SCS)
#    
#    return S.value