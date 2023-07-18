#Functions for quantum metrology

import numpy as np
import scipy as sp
from numpy.polynomial.hermite import hermval
from numpy import linalg as LA
from scipy.special import comb,hermite,factorial
from sympy import *
from scipy import linalg
from scipy.linalg import expm, sinm, cosm, expm_multiply
from scipy import sparse
from scipy.sparse import linalg as las
from scipy.sparse import csr_matrix,coo_matrix,csc_matrix
from scipy.optimize import curve_fit
import matplotlib as mpl
from matplotlib import pyplot as plt

def fastqfiloss_fullrank(ang,state,n):
    #Compute spectral formula for QFI of interferometer using a full rank state approximation to a diagonal state
    #state should be n+1 x n+1 diagonal matrix
    rr=state
    size=n+1
    small=(10**-10)
    dd=((1-small)*rr)+ (small*(1/size)*np.eye(size))
    specsy=np.diag(dd)
    #Rotate the eigenvectors
    yy=sg.JYm_sparse(size-1)
    rotveclist=[]
    for j in range(len(specsy)):
        vv=np.zeros(size).astype(np.complex128)
        vv[j]=1
        rotveclist.append(expm_multiply(-(1j)*ang*yy,vv))
    
    
    
    delrho1y=-(1j)*( yy@expm_multiply( -(1j)*ang*yy, expm_multiply(-(1j)*ang*yy,rr).conj().T))
    delrho2y=(1j)*expm_multiply( -(1j)*ang*yy, expm_multiply(-(1j)*ang*yy,rr).conj().T)@yy
    delrhoy=delrho1y+delrho2y
    
    
    fishy=[]
    for i in range(size):
        for j in range(size):
            ffy=(1/(specsy[i]+specsy[j]))
            ggy=np.conj(rotveclist[i])@delrhoy@rotveclist[j]
            fishy.append(ffy*(np.abs(ggy)**2))
    ffy=2*np.sum(fishy)
    
    return (1/n)*np.real(ffy)
