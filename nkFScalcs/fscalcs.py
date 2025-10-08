from .fsconst import *
import numpy as np

def P2Q(P):
    n, m = P.shape
    if m != n:
        raise ValueError("Input matrix P must have shape (n, n)")

    Q = np.zeros((n, m))
    for i in range(n):
        if i == 0:
            Q[0, 0] = np.sum(P[0, :-1])
        else:
            Q[i, :i] = P[i, :i]
            Q[i, i] = np.sum(P[i, i:-1])
    Q[:,m-1] = P[:,m-1]
    return Q

def matpower(P,n):
    return np.linalg.matrix_power(P, n)

def set_MVc(TtM,TtDef,RcFactors,RFRs):
    return RcFactors ** ((TtM - TtDef)/15) * (1+RFRs[TtDef-1])**TtDef

def set_CoD_M_d(TtM,TtDef,RcFactors,RFRs):
    MVc = set_MVc(TtM,TtDef,RcFactors,RFRs)
    CoDM = np.zeros((num_CQSs,num_CQSs))
    for i in range(num_CQSs):
        for j in range(i+1,num_CQSs):
            CoDM[i,j] = MVc[i]-MVc[j]
    return CoDM

def set_CMm(T,CoDM,MVc,RR):
    n, m = T.shape
    l, o = CoDM.shape
    q = MVc.flatten().shape[0]
    if (m != n) or (l != o) or (l != q) or (n != l + 1):
        raise ValueError("Some dimensions are not right - when trying to compute CMm;",n,m,l,o,q)
    CMm = np.zeros((n,n))
    CMm[:-1,-1] = (1-RR) * T[:-1,-1] * MVc.flatten()
    CMm[:-1,:-1] = CoDM * T[:-1,:-1]
    return CMm

def MatrixCMx(RcFactors,BondTtm,RFRs,P,RR):
    Q = P2Q(P)
    disc_vector = (1 + RFRs) ** (-(np.arange(len(RFRs)) + 0.5))
    RelRFR = RFRs[BondTtm-1]
    mymat = np.zeros((num_CQSs,BondTtm))
    mymat_PD_T = np.zeros((num_CQSs,BondTtm))
    mymat_PD_Q = np.zeros((num_CQSs,BondTtm))
    for i in range(BondTtm):
        MVc = set_MVc(BondTtm,i+1, RcFactors, RFRs)
        CoDM = set_CoD_M_d(BondTtm,i+1,RcFactors,RFRs)
        CMm = set_CMm(P,CoDM,MVc,RR)
        FinVec_CoD = (matpower(Q,i) @ (CMm @ CoD_vector))[:-1]
        FinVec_PD_T = (matpower(P,i) @ (CMm @ PD_vector))[:-1]
        FinVec_PD_Q = (matpower(Q,i) @ (CMm @ PD_vector))[:-1]
        mymat[:,i] = FinVec_CoD
        mymat_PD_T[:,i] = FinVec_PD_T
        mymat_PD_Q[:,i] = FinVec_PD_Q
     
    mymat2 = mymat * disc_vector[:BondTtm]    
    mymatCoD1 = (1+RelRFR) * (1-np.sum(mymat2,axis=1))**(-1/BondTtm) -1 - RelRFR
    mymatT = mymat_PD_T * disc_vector[:BondTtm]    
    mymatPD_T = (1+RelRFR) * (1-np.sum(mymatT,axis=1))**(-1/BondTtm) -1 - RelRFR
    mymatQ = mymat_PD_Q * disc_vector[:BondTtm]    
    mymatPD_Q = (1+RelRFR) * (1-np.sum(mymatQ,axis=1))**(-1/BondTtm) -1 - RelRFR
    mymatCoDF = np.maximum(0, mymatCoD1 - (mymatPD_T - mymatPD_Q))
    return mymatCoD1, mymatPD_T, mymatPD_Q, mymatCoDF  # mymatPD_T and mymatCoDF are the important ones that are used

def CreateFSTables(RcFactors,FinalMaturity,RFRs,P,RecovRate, clipat = 0.98):
    mymat_PD_prob = np.zeros((num_CQSs,FinalMaturity))
    FS_PD = np.zeros((num_CQSs,FinalMaturity))
    FS_CoD = np.zeros((num_CQSs,FinalMaturity))
    for i in range(FinalMaturity):
        dummy1, FS_PD[:,i], dummy2, FS_CoD[:,i] = MatrixCMx(RcFactors,i+1,RFRs,P,RecovRate)
        mymat_PD_prob[:,i] = (matpower(P,i+1))[:-1,-1]

    FS_PD = np.clip(FS_PD, None, clipat)  
    FS_CoD = np.clip(FS_CoD, None, clipat)

    return mymat_PD_prob.T, FS_PD.T, FS_CoD.T
