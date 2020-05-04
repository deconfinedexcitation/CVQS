import numpy as np
from tensor_network_functions import*
from scipy.linalg import expm, sinm, cosm
from numpy import linalg as LA


## Local operators
I = np.identity(2)
Z = np.zeros((2,2))
Sz = np.array([[1, 0],
             [0, -1]])
#Next is (-i/2)\sigma_{y}
miov2Sy = np.array([[0, -0.5],
             [0.5, 0]])
#(-i/2)\sigma_{x}
miov2Sx = 0.5*np.array([[0, -1j],
             [-1j, 0]])

Pz = np.array([[1, 0],
             [0, 0]])
Po = np.array([[0, 0],
             [0, 1]])

def expsx(th):
    return np.array([[np.cos(th/2),-(1j)*np.sin(th/2)],[-(1j)*np.sin(th/2),np.cos(th/2)]])
def expsy(th):
    return np.array([[np.cos(th/2),-np.sin(th/2)],[np.sin(th/2),np.cos(th/2)]])

def expsxlayer(N,th):
    ulist2=[ np.array( [ [  expsx(th)  ] ] ) ]
    for j in range(1,N):
        ulist2+=[ np.array( [ [  expsx(th)  ] ] ) ]
    return ulist2

def expsylayer(N,th):
    ulist2=[ np.array( [ [  expsy(th)  ] ] ) ]*N
    return ulist2

def idlayer(N):
    ulist2=[ np.array( [ [  I  ] ] ) ]
    for j in range(1,N):
        ulist2+=[ np.array( [ [  I  ] ] ) ]
    return ulist2

def zlayer(N):
    ulist2=[ np.array( [ [  Sz  ] ] ) ]
    for j in range(1,N):
        ulist2+=[ np.array( [ [  Sz  ] ] ) ]
    return ulist2

def CZti(N):
    Q=[np.array([[Pz,Po]])]
    for j in range(1,N-1):
        Q+=[np.array([[Pz,Po  ],
                 [Pz,-Po ]])]
    Q+=[np.array([[I],[Sz]])]
    return Q

def CZtiparam(th,N):
    #\prod_{j=1}^{n-1}CZ_{j,j+1} with MPO of bond dimension 4. CXXVI p.88
    Q=[np.array([[np.sqrt((1/2)*np.cos(th))*I,np.sqrt((1/2)*np.cos(th))*I,np.sqrt((1j)*np.sin(th))*Pz,np.sqrt((1j)*np.sin(th))*Po]])]
    for j in range(1,N-1):
        Q+=[np.array([[(1/2)*np.cos(th)*I,(1/2)*np.cos(th)*I,(1/2)*np.sqrt((1j)*np.sin(2*th))*Pz, (1/2)*np.sqrt((1j)*np.sin(2*th))*Po ],
                 [(1/2)*np.cos(th)*I,(1/2)*np.cos(th)*I,(1/2)*np.sqrt((1j)*np.sin(2*th))*Pz, (1/2)*np.sqrt((1j)*np.sin(2*th))*Po ],
                 [(1/2)*np.sqrt((1j)*np.sin(2*th))*I,(1/2)*np.sqrt((1j)*np.sin(2*th))*I, (1j)*np.sin(th)*Pz, (1j)*np.sin(th)*Po ],
                 [(1/2)*np.sqrt((1j)*np.sin(2*th))*Sz,(1/2)*np.sqrt((1j)*np.sin(2*th))*Sz, (1j)*np.sin(th)*Pz, -(1j)*np.sin(th)*Po ] ])]
    Q+=[np.array([[np.sqrt((1/2)*np.cos(th))*I],[np.sqrt((1/2)*np.cos(th))*I],[np.sqrt((1j)*np.sin(th))*I],[np.sqrt((1j)*np.sin(th))*Sz]])]
    return Q

def CZtiparam2(th,N):
    #\prod_{j=1}^{n-1}CZ_{j,j+1} with MPO of bond dimension 2.
    #Note that e^{-\theta CZ}=1\otimes 1 + (e^{i\theta}-1)\ket{1}\bra{1}\otimes \ket{1}\bra{1}
    Q=[np.array([[I,Po]])]
    for j in range(1,N-1):
        Q+=[np.array([[I,Po],
                 [(np.exp((1j)*th)-1)*Po,(np.exp((1j)*th)-1)*Po] ])]
    Q+=[np.array([[I],[(np.exp((1j)*th)-1)*Po]])]
    return Q
