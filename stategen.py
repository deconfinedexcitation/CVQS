##test
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

#This file contains functions that do things like: 1. output generators of unitary operators, 2. output common continuous-variable quantum states.

#C XXIV p.129

#The QuantumOptics.jl package (based on QuTiP) calculates the coherent state from the unitary displacement operator.
# Xanadu codes the unitary squeezing, unitary displacement, unitary two-mode squeezing operators, etc. in
# fock_gradients.py in their package TheWalrus. They use a recursive method from Quesada "Fast optimization of parameterized quantum
# optical circuits"


#Note that specifying the Fock amplitudes of the coherent state either iteratively or explicitly via np.float64( np.exp( -((alpha**2)/2) - np.log(np.sqrt(factorial(j))) + (  j*np.log(alph)  )  )  ) gives error or erroneous results for large energy.




#Generator of real displacements

def dispham(alph,cut):
    Hrow=[]
    Hcol=[]
    Hdata=[]
    for k in range(cut):
        for j in range(cut):
            if j==k+1:
                Hcol.append(k)
                Hrow.append(j)
                Hdata.append(alph*np.sqrt(k+1))
    Hrow=np.int_(np.asarray(Hrow))
    Hcol=np.int_(np.asarray(Hcol) )
    Hdata=np.asarray(Hdata)
    H=coo_matrix((Hdata, (Hrow, Hcol)), shape=(cut, cut)).tocsr()
    H=H-np.transpose(H)
    return H

#Generator of real squeezing

def sqham(r,cut):
    Hrow=[]
    Hcol=[]
    Hdata=[]
    for k in range(cut):
        for j in range(cut):
            if j==k+2:
                Hcol.append(k)
                Hrow.append(j)
                Hdata.append(-(r/2)*np.sqrt((k+1)*(k+2)))
    Hrow=np.int_(np.asarray(Hrow))
    Hcol=np.int_(np.asarray(Hcol) )
    Hdata=np.asarray(Hdata)
    H=coo_matrix((Hdata, (Hrow, Hcol)), shape=(cut, cut)).tocsr()
    H=H-np.transpose(H)
    return H
    
#Kerr gate in Fock basis.

def kerr(time,kappa, cutoff, dtype=np.complex128):  
    r"""Calculates the matrix elements of the Kerr gate e^{-it\kappa (a^{*}a)^{2}} using a recurrence relation.

    Args:
        r (float): time
        kappa (float): nonlinearity
        cutoff (int): Fock ladder cutoff
        dtype (data type): Specifies the data type used for the calculation

    Returns:
        array[complex]: matrix representing the single mode Kerr evolution
    """
    nlin=np.exp(-(1j) * time*kappa)
    S = np.zeros((cutoff, cutoff), dtype=dtype)
    S[0,0]=1
    for n in range(cutoff-1):
        S[n+1,n+1]=(nlin**((2*n)+1))*S[n,n]
    return S

#Coherent state from generator
def coh(ener):
    cut=np.ceil(10*(ener))
    cut=np.int(cut)
    ms=np.zeros(cut)
    ms[0]=1
    ms=sp.sparse.linalg.expm_multiply(   dispham(np.sqrt(ener),cut)   ,ms)
    return ms

#Coherent state by recursion

def cohrec(ener,cutoff):
    vv=np.zeros(cutoff)
    vv[0]=1
    for j in range(cutoff-1):
        vv[j+1]=np.sqrt(ener)*(1/np.sqrt(j+1))*vv[j]
    st=np.exp(-(1/2)*ener)*vv
    return st

#Coherent state via quantum central limit
def cohcl(ener,N):
    Hrow=[]
    Hcol=[]
    Hdata=[]
    for k in range(N+1):
        for j in range(N+1):
            if j==k+1:
                Hcol.append(k)
                Hrow.append(j)
                Hdata.append((np.sqrt(ener)/np.sqrt(N))*np.sqrt((k+1)*(N-k)))
            elif j==k:
                Hcol.append(k)
                Hrow.append(j)
                Hdata.append(-(N/2)*np.log(1+(ener/N)))
    Hrow=np.int_(np.asarray(Hrow))
    Hcol=np.int_(np.asarray(Hcol) )
    Hdata=np.asarray(Hdata) 
    H=coo_matrix((Hdata, (Hrow, Hcol)), shape=(N+1, N+1)).tocsr()
    b=np.zeros(N+1)
    b[0]=1
    rr=sp.sparse.linalg.expm_multiply(   H   ,b)
    #rr=rr/np.max(np.abs(rr))
    #rr=rr/(np.sqrt((np.sum(np.square(rr)))))
    return rr

#Squeezed state

def sq(ener):
    cut=np.ceil(10*(ener))
    cut=np.int(cut)
    ms=np.zeros(cut)
    ms[0]=1
    ms=sp.sparse.linalg.expm_multiply(  sqham(np.arcsinh(np.sqrt(ener)),cut)   ,ms)
    return ms

#Displaced squeezed state with energy distribution as in ``Maximal trace distance between isoenergetic...'' and ``Linear bosonic channels defined by...''

def dispsq(ener):
    cut=np.ceil(10*(ener))
    cut=np.int(cut)
    d=(2*ener)+1
    r=np.sqrt(((ener**2)+ener)/((2*ener)+1))
    w=(1/2)*np.log((2*ener)+1)
    ms=np.zeros(cut)
    ms[0]=1
    ms=sp.sparse.linalg.expm_multiply( dispham(r,cut), sp.sparse.linalg.expm_multiply(   sqham(w,cut)   ,ms) ) 
    ms=ms/np.linalg.norm(ms)
    return ms


#Correct direct specification of superposition of maximally distant isoenergetic Gaussian states
def maxsupstate_corr(ener):
    cut=np.ceil(10*(ener))
    cut=np.int(cut)
    d=(2*ener)+1
    r=np.sqrt(((ener**2)+ener)/((2*ener)+1))
    w=(1/2)*np.log((2*ener)+1)
    ms=np.zeros(cut)
    ms[0]=1
    ms=sp.sparse.linalg.expm_multiply( dispham(r,cut), sp.sparse.linalg.expm_multiply(   sqham((1/2)*np.log(d),cut)   ,ms) ) + sp.sparse.linalg.expm_multiply( dispham(-r,cut), sp.sparse.linalg.expm_multiply(   sqham((1/2)*np.log(d),cut)   ,ms) )
    ms=ms/np.linalg.norm(ms)
    return ms



#Even cat
def evencat(ener):
    cut=np.ceil(10*(ener))
    cut=np.int(cut)
    ms=np.zeros(cut)
    ms[0]=1
    ms=sp.sparse.linalg.expm_multiply(   dispham(np.sqrt(ener),cut)   ,ms) + sp.sparse.linalg.expm_multiply(   dispham(-np.sqrt(ener),cut)   ,ms)
    ms=ms/np.linalg.norm(ms)
    return ms


# SU(2) coherent states. Uses stereographic projection from south pole.
# Opposite of Perelomov's convention.
# o.n.b. |N-k,k> for k=0,1,\ldots , N

def Jplus(n):
    Jp=np.zeros((n+1,n+1))
    for k in range(0, np.int(n)):
        Jp[k][k+1]=np.sqrt((n-k)*(k+1))
    return np.array(Jp)
    
def Jminus(n):    
    return np.transpose(np.array(Jplus(n)))
    
def JZm(n):
    Jz=np.zeros((n+1,n+1))
    for k in range(np.int(n)+1):
        Jz[k][k]=(1/2)*(n-(2*k))
    return np.array(Jz)
    
def JYm(n):
    #J_{y} in spin-n/2 rep'n
    xx=Jplus(n)
    Jym=(1/(2*(1.0*1j)))*(np.array(xx)-np.transpose(np.array(xx)))
    return Jym
    
    
def JXm(n):
    #J_{x} in spin-n/2 rep'n
    xx=Jplus(n)
    Jxm=(1/2)*(np.array(xx)+np.transpose(np.array(xx)))
    return Jxm

def JYm2(n):
    #This is (1j)*JYm
    xx=Jplus(n)
    Jym=(1/2)*(np.array(xx)-np.transpose(np.array(xx)))
    return Jym
    
def su2cs(phi,thet,n):
    invec=np.zeros(n+1)
    invec[0]=1
    f=expm(np.exp(-(1j)*phi)*np.tan(thet/2)*Jminus(n))@np.transpose(invec)
    f=f/np.sqrt(np.abs(np.dot(np.conj(f),f)))
    return f
    
# SU(2) coherent states. Uses stereographic projection from north pole.
# o.n.b. |k,N-k> for k=0,1,\ldots ,N

def Jplus_north(n):
    Jp=np.zeros((n+1,n+1))
    for k in range(0, np.int(n)):
        Jp[k+1][k]=np.sqrt((n-k)*(k+1))
    return np.array(Jp)
    
def Jminus_north(n):    
    return np.transpose(np.array(Jplus_north(n)))

def Jz_north(n):
    Jz=np.zeros((n+1,n+1))
    for k in range(np.int(n)+1):
        Jz[k][k]=(1/2)*((2*k)-n)
    return np.array(Jz)
    
def JYm_north(n):
    #J_{y} in spin-n/2 rep'n
    xx=Jplus_north(n)
    Jym=(1/(2*(1.0*1j)))*(np.array(xx)-np.transpose(np.array(xx)))
    return Jym
    
def JXm_north(n):
    #J_{x} in spin-n/2 rep'n
    xx=Jplus_north(n)
    Jxm=(1/2)*(np.array(xx)+np.transpose(np.array(xx)))
    return Jxm

def JYm2_north(n):
    #This is (1j)*JYm_north
    xx=Jplus_north(n)
    Jym=(1/2)*(np.array(xx)-np.transpose(np.array(xx)))
    return Jym
    
def su2cs_north(phi,thet,n):
    invec=np.zeros(n+1)
    invec[n]=1
    f=expm(np.exp(-(1j)*phi)*np.tan(thet/2)*Jplus_north(n))@np.transpose(invec)
    f=f/np.sqrt(np.abs(np.dot(np.conj(f),f)))
    return f
