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
        rotveclist.append(las.expm_multiply(-(1j)*ang*yy,vv))
    
    
    
    delrho1y=-(1j)*( yy@las.expm_multiply( -(1j)*ang*yy, las.expm_multiply(-(1j)*ang*yy,dd).conj().T))
    delrho2y=(1j)*las.expm_multiply( -(1j)*ang*yy, las.expm_multiply(-(1j)*ang*yy,dd).conj().T)@yy
    delrhoy=delrho1y+delrho2y
    
    
    fishy=0
    #Instead of using the rotated vectors, just multiply
    #delrhoy by sparse matrices, then access matrix elements.
    aaa=las.expm_multiply((1j)*ang*yy,las.expm_multiply((1j)*ang*yy,delrhoy).conj().T)
    for i in range(size+1):
        #Diagonal contribution
        ffy=(1/(2*specsy[i]))
        ggy=aaa[i,i]
        fishy+= ffy*(np.abs(ggy)**2)
    for i in range(size+1):
        for j in range(i+1,size+1):
            #Off-diagonal contributions
            ffy=(1/(specsy[i]+specsy[j]))
            ggy=aaa[i,j]
            fishy+= 2*ffy*(np.abs(ggy)**2)
            ffy=2*fishy
    
    return (1/n)*np.real(fishy)
