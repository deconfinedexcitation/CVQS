import numpy as np
import scipy as sp
from numpy import linalg
from numpy.polynomial.hermite import hermval
from scipy.special import comb,hermite
from scipy.special import factorial
from scipy.linalg import expm, sinm, cosm
from scipy import sparse
from scipy.sparse import csr_matrix,coo_matrix
from scipy.optimize import curve_fit
import matplotlib as mpl
from matplotlib import pyplot as plt
from CVQS/stategen import *

##Qubit gates

def qubgate(xx):
    #Use xx[0] the rotation angle in -np.pi,np.pi
    #Use xx[1] the polar angle in 0,np.pi
    #Use xx[2] the azimuth in -np.pi,np.pi
    jxop=JXm(1)
    jyop=JYm(1)
    jzop=JZm(1)
    ff=expm(-(1j)*xx[0]*((np.cos(xx[1])*jzop)+(np.sin(xx[1])*(np.cos(xx[2])*jxop + np.sin(xx[2])*jyop))))
    return ff
def zyz(xx):
    #ZYZ Euler angle decomposition for e^{-i\theta n\cdot J}
    #for e^{-i\theta n\cdot J} specified in qubgate
    jxop=JXm(1)
    jyop=JYm(1)
    jzop=JZm(1)
    hh1=expm(-(1j)*alpha(xx)*jzop)@expm(-(1j)*beta(xx)*jyop)@expm(-(1j)*gamma(xx)*jzop)
    return hh1

##Gates for qudit quantum computation
##See Sanders ``Qudits and high-dimensional...''

def cshift(d):
    #Create vectors \ket{j}
    pp=[list(np.zeros(d)) for j in range(d)]
    for j in range(d):
        pp[j][j]=1
    pp=np.array(pp)
    shf=np.zeros((d**2,d**2))
    for j in range(d):
        for k in range(d):
            shf=shf+np.outer( np.kron(pp[j],pp[np.mod((k+j),d)]),np.kron(pp[j],pp[k]))
    return shf

def Z(z,d):
    gg=[]
    for j in range(d):
        gg.append(np.exp((1j)*2*np.pi*z*j/d))
    return np.diag(gg)

def X(x,d):
    vv=np.zeros((d,d))
    for k in range(d):
        for kp in range(d):
            if (kp==np.mod(k+x,d)):
                vv[kp,k]=1
    return vv

def maxentangled(d):
    vv=np.zeros(d**2)
    for j in range(d):
        vv1=np.zeros(d)
        vv1[j]=1
        vv=vv+np.kron(vv1,vv1)
    return (1/np.sqrt(d))*vv

def bell(x,z,d):
    return np.kron(X(x,d),Z(z,d))@maxentangled(d)
