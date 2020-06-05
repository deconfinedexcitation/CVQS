import numpy as np
from tensor_network_functions import*
from scipy.linalg import expm, sinm, cosm
from numpy import linalg as LA


## Local operators
I = np.identity(2)
Z = np.zeros((2,2))
Sz = np.array([[1, 0],
             [0, -1]])
Sx = np.array([[0, 1],
             [1, 0]])
Sy = np.array([[0, -(1j)],
             [(1j), 0]])
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
    #e^{i\theta \prod_{j=1}^{n-1}CZ_{j,j+1} } with MPO of bond dimension 4. CXXVI p.88
    Q=[np.array([[np.sqrt((1/2)*np.cos(th))*I,np.sqrt((1/2)*np.cos(th))*I,np.sqrt((1j)*np.sin(th))*Pz,np.sqrt((1j)*np.sin(th))*Po]])]
    for j in range(1,N-1):
        Q+=[np.array([[(1/2)*np.cos(th)*I,(1/2)*np.cos(th)*I,(1/2)*np.sqrt((1j)*np.sin(2*th))*Pz, (1/2)*np.sqrt((1j)*np.sin(2*th))*Po ],
                 [(1/2)*np.cos(th)*I,(1/2)*np.cos(th)*I,(1/2)*np.sqrt((1j)*np.sin(2*th))*Pz, (1/2)*np.sqrt((1j)*np.sin(2*th))*Po ],
                 [(1/2)*np.sqrt((1j)*np.sin(2*th))*I,(1/2)*np.sqrt((1j)*np.sin(2*th))*I, (1j)*np.sin(th)*Pz, (1j)*np.sin(th)*Po ],
                 [(1/2)*np.sqrt((1j)*np.sin(2*th))*Sz,(1/2)*np.sqrt((1j)*np.sin(2*th))*Sz, (1j)*np.sin(th)*Pz, -(1j)*np.sin(th)*Po ] ])]
    Q+=[np.array([[np.sqrt((1/2)*np.cos(th))*I],[np.sqrt((1/2)*np.cos(th))*I],[np.sqrt((1j)*np.sin(th))*I],[np.sqrt((1j)*np.sin(th))*Sz]])]
    return Q

def CZtiparam2(th,N):
    #e^{i\theta \prod_{j=1}^{n-1}CZ_{j,j+1} } with MPO of bond dimension 2.
    #Note that e^{-\theta CZ}=1\otimes 1 + (e^{i\theta}-1)\ket{1}\bra{1}\otimes \ket{1}\bra{1}
    Q=[np.array([[I,Po]])]
    for j in range(1,N-1):
        Q+=[np.array([[I,Po],
                 [(np.exp((1j)*th)-1)*Po,(np.exp((1j)*th)-1)*Po] ])]
    Q+=[np.array([[I],[(np.exp((1j)*th)-1)*Po]])]
    return Q




def isingint(n):
    #Ising interaction on (C^{2})^{\otimes n} with p.b.c.. Not an MPO
    zz=-(1/2)*np.kron(Sz,Sz)
    c=np.kron(zz,np.identity(2**(n-2)))
    for j in range(1,n-2):
        c+=np.kron( np.kron(np.identity(2**j),zz),np.identity(2**(n-j-2)) )
    c=c+np.kron(np.identity(2**(n-2)),zz)
    #The (n,1) term for p.b.c.
    c=c+((-1/2)*np.kron(np.kron(Sz,np.identity(2**(n-2))),Sz))
    return c




def ising_int_mpo(n):
        #Ising interaction MPO. See Pirvu or Chan ``A simplified and improved...''
        #\sum_{j=1}^{n-1}\sigma_{z}^{(j)}\otimes \sigma_{z}^{(j+1)}
    loc1 =np.array([[Z, Sz, I   ]])
    loc=np.array([[I,  Z, Z  ],\
                  [Sz  , Z, Z],\
                  [Z, Sz, I]])
    loc2 =np.array([[I], [Sz], [Z] ])
    if n>2:
        loccost=[loc1]+([loc]*(np.int(n)-2))+[loc2]
    else:
        loccost=[loc1]+[loc2]
    return loccost

def ising_int_dyn(gamma,n):
    #MPO for e^{-i\gamma \sum_{j=1}^{n-1}Z_{j}\otimes Z_{j+1}}
    #CXXVI p.74
    Q1 =np.array([[np.cos(gamma)*I, -(1j)*np.sin(gamma)*Sz   ]])
    Q=np.array([[np.cos(gamma)*I,  -(1j)*np.sin(gamma)*Sz  ],
             [np.cos(gamma)*Sz  , -(1j)*np.sin(gamma)*I]])
    Q2=np.array([[I],[Sz]])
    driv=[Q1]+([Q]*(n-2))+[Q2]
    return driv
