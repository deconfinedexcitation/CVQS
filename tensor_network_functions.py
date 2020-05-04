import numpy as np
import scipy
import scipy.sparse.linalg
import scipy.sparse as sparse
import math
from copy import deepcopy
 
# MPS A-matrix is a 3-index tensor, A[s,i,j]
#    s
#    |
# i -A- j
#
# [s] acts on the local Hilbert space
# [i,j] act on the virtual edges
 
# MPO W-matrix is a 4-index tensor, W[s,t,i,j]
#     s
#     |
#  i -W- j
#     |
#     t
#
# [s,t] act on the local Hilbert space,
# [i,j] act on the virtual edges


 
## tensor contraction from the right hand side
##  -+     -A--+
##   |      |  |
##  -F' =  -W--F
##   |      |  |
##  -+     -B--+
def contract_from_right(W, A, F, B):
    # the einsum function doesn't appear to optimize the contractions properly,
    # so we split it into individual summations in the optimal order
    #return np.einsum("abst,sij,bjl,tkl->aik",W,A,F,B, optimize=True)
    Temp = np.einsum("sij,bjl->sbil", A, F)
    Temp = np.einsum("sbil,abst->tail", Temp, W)
    return np.einsum("tail,tkl->aik", Temp, B)

## tensor contraction from the left hand side
## +-     +--A-
## |      |  |
## E'- =  E--W-
## |      |  |
## +-     +--B-
def contract_from_left(W, A, E, B):
    # the einsum function doesn't appear to optimize the contractions properly,
    # so we split it into individual summations in the optimal order
    # return np.einsum("abst,sij,aik,tkl->bjl",W,A,E,B, optimize=True)
    Temp = np.einsum("sij,aik->sajk", A, E)
    Temp = np.einsum("sajk,abst->tbjk", Temp, W)
    return np.einsum("tbjk,tkl->bjl", Temp, B)

## Function the evaluate the expectation value of an MPO on a given MPS
## <A|MPO|B>
def Expectation(AList, MPO, BList):
    E = [[[1]]]
    for i in range(0,len(MPO)):
        E = contract_from_left(MPO[i], AList[i], E, BList[i])
    return E[0][0][0]

# 'vertical' product of MPO W-matrices
#        |
#  |    -W-
# -R- =  |
#  |    -X-
#        |
def product_W(W, X):
    return np.reshape(np.einsum("abst,cdtu->acbdsu", W, X), [W.shape[0]*X.shape[0],
                                                             W.shape[1]*X.shape[1],
                                                             W.shape[2],X.shape[3]])
 
def product_MPO(M1, M2):
    assert len(M1) == len(M2)
    Result = []
    for i in range(0, len(M1)):
        Result.append(product_W(M1[i], M2[i]))
    return Result

#TJV 01/2020 Implement MPOMPS, viz., application of MPO to MPS
#
# 'vertical' product of MPO and MPS
#        |
#  |    -W-
# -A'- = |
#       -A-
#      


def product_A(W,A):
    #Recall that the 0 index of A is the physical index, so it gets contracted with the 
    #second physical index of W. The 1 and 2 indices of A are virtual,
    return np.reshape(np.einsum("abst,tdu->sadbu", W, A), [W.shape[2],W.shape[0]*A.shape[1],
                                                             W.shape[1]*A.shape[2]
                                                             ])

def MPOMPS(M1, M2):
    assert len(M1) == len(M2)
    Result = []
    for i in range(0, len(M1)):
        Result.append(product_A(M1[i], M2[i]))
    return Result
