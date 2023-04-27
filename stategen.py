##test
import numpy as np
import scipy as sp
from numpy.polynomial.hermite import hermval
from scipy.special import comb,hermite,factorial
from scipy import linalg
from scipy.linalg import expm, sinm, cosm
from scipy import sparse
from scipy.sparse import linalg as las
from scipy.sparse import csr_matrix,coo_matrix,csc_matrix
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

def beamsplit_cs(z1,z2,N):
    vv1=sg.cohrec(z1**2,cutoff)
    vv2=sg.cohrec(z2**2,cutoff)
    #Need np.complex128 data type since fock_gradients.beamsplitter is given as such
    vv1=np.array(vv1,dtype=np.complex128)
    vv2=np.array(vv2,dtype=np.complex128)
    bb=np.zeros((cutoff,cutoff),dtype=np.complex128)
    #Eq.(78) of "Fast optimization..."
    for j in range(cutoff):
        for k in range(cutoff):
            for l in range(cutoff):
                for s in range(cutoff):
                    bb[j][k]=bb[j][k]+(ccc[j][k][l][s]*vv1[l]*vv2[s])
## A sum over l and s is not necessary due to number conservation.
## See (77) of "Fast optimization..."
#         low=np.max([1+j+k-cutoff,0])
#         high=np.min([j+k,cutoff-1])+1
#         for r in range(low,high,1):
#             bb[j][k]+=ccc[j][k][r][j+k-r]*vv1[r]*vv2[j+k-r]
    #Return to tensor product form
    rr=np.reshape(bb,(cutoff**2))
    return rr

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
    
def Jplus_sparse(n):
    Hrow=[]
    Hcol=[]
    Hdata=[]
    for k in range(n+1):
        for j in range(n+1):
            if j==k+1:
                Hrow.append(k)
                Hcol.append(j)
                Hdata.append(np.sqrt((n-k)*j))
    Hrow=np.int_(np.asarray(Hrow))
    Hcol=np.int_(np.asarray(Hcol) )
    Hdata=np.asarray(Hdata)
    H=coo_matrix((Hdata, (Hrow, Hcol)), shape=(n+1, n+1)).tocsr()
    return H

def Jminus_sparse(n):
    return Jplus_sparse(n).transpose()
    
def JYm_sparse(n):
    aa=Jplus_sparse(n)-Jminus_sparse(n)
    aa=aa/(2*(1j))
    return aa
    
def JZm_sparse(n):
    Hrow=[]
    Hcol=[]
    Hdata=[]
    for k in range(n+1):
        Hrow.append(k)
        Hcol.append(k)
        Hdata.append((1/2)*(n-(2*k)))
    Hrow=np.int_(np.asarray(Hrow))
    Hcol=np.int_(np.asarray(Hcol) )
    Hdata=np.asarray(Hdata)
    H=coo_matrix((Hdata, (Hrow, Hcol)), shape=(n+1, n+1)).tocsr()
    return H
    
def prodx(n):
    ## X^{\otimes n}
    aa=np.zeros((n+1, n+1))
    aa=np.diag(np.ones(n+1))
    aa=np.fliplr(aa)
    return aa

def prody(n):
    ## Y^{\otimes n}
    aa=np.zeros((n+1, n+1))
    vv=[]
    for j in range(n+1):
        vv.append(((1j)**n)*((-1)**j))
    aa=np.diag(vv)
    aa=np.fliplr(aa)
    return aa
    
def prodz(n):
    ## Z^{\otimes n}
    aa=np.zeros((n+1, n+1))
    vv=[]
    for j in range(n+1):
        vv.append(((-1)**n))
    aa=np.diag(vv)
    return aa
    
def su2cs(phi,thet,n):
    invec=np.zeros(n+1)
    invec[0]=1
    f=expm(np.exp(-(1j)*phi)*np.tan(thet/2)*Jminus(n))@np.transpose(invec)
    f=f/np.sqrt(np.sum(np.abs(f)**2))
    return f
    
def su2cs_sparse(thet,n):
    irow=[]
    icol=[]
    idata=[]
    for k in range(n+1):
        irow.append(k)
        icol.append(0)
        if k==0:
            idata.append(1)
        else:
            idata.append(0)
    irow=np.int_(np.asarray(irow))
    icol=np.int_(np.asarray(icol) )
    idata=np.asarray(idata)
    invec=coo_matrix((idata, (irow, icol)), shape=(n+1,1)).tocsc()
    
    
    #Split the application of the rotation into many steps
    #to avoid calculation of a large normalization constant
    op=las.expm(-(1j)*thet*JYm_sparse(n)/(n/10))
    f=invec
    for j in range(int(n/10)):
        f=op@(f/las.norm(f))
    g=np.array(f.transpose().todense())[0]
    return g

def su2cs_plus(x,n):
    invec=np.zeros(n+1)
    invec[n]=1
    f=expm(x*Jplus(n))@invec
    f=f/np.sqrt(np.abs(np.dot(np.conj(f),f)))
    return f
    
## CV Gaussian states

def symp(m):
    #Symplectic matrix in Holevo ordering of canonical operators
    ff=np.array([[0,1],[-1,0]])
    for j in range(m-1):
        ff=linalg.block_diag(ff,np.array([[0,1],[-1,0]]))
    return ff

def hol_to_qp(m):
    ## (q_{1},...,q_{M},p_{1},...,p_{M})=(q_{1},p_{1},...,q_{M},p_{M})A
    cc=np.zeros(2*m)
    cc[0]=1
    vvv=[np.roll(cc,j) for j in range(2*m)]
    rrr=[]
    for j in range(m):
        rrr.append(vvv[2*j])
    for j in range(m):
        rrr.append(vvv[(2*j)+1])
    A=np.transpose(np.array(rrr))
    return A
    
#Cat states, compass states, phase states, twin Fock states

def phasestate_z(thet,n):
    vv=np.zeros(n+1)
    for j in range(n+1):
        aa=np.zeros(n+1)
        aa[j]=1
        vv=vv + np.exp((1j)*((n/2)-j)*thet)*aa
    return vv/np.sqrt(np.sum(np.abs(vv)**2))

##It is better to code up cat states using rotations
## than by taking superpositions of eigenvectors of
## output by linalg.eig. This is because a multiplicative phase
## of the eigenvector is not fixed.

def xcat(n):
    vv=np.zeros(n+1)
    vv[0]=1
    ww=np.zeros(n+1)
    ww[n]=1
    state=expm(-(1j)*(np.pi/2)*JYm(n))@((vv+ (((-1)**n)*ww))/np.sqrt(2))
    return state

def xminuscat(n):
    vv=np.zeros(n+1)
    vv[0]=1
    ww=np.zeros(n+1)
    ww[-1]=1
    state=expm(-(1j)*(np.pi/2)*JYm(n))@((vv-(((-1)**n)*ww))/np.sqrt(2))
    return state

def ycat(n):
    vv=np.zeros(n+1)
    vv[0]=1
    ww=np.zeros(n+1)
    ww[-1]=1
    state=expm((1j)*(np.pi/2)*JXm(n))@((vv+(((-(1j))**n)*ww))/np.sqrt(2))
    return state

def yminuscat(n):
    vv=np.zeros(n+1)
    vv[0]=1
    ww=np.zeros(n+1)
    ww[-1]=1
    state=expm((1j)*(np.pi/2)*JXm(n))@((vv-(((-(1j))**n)*ww))/np.sqrt(2))
    return state

def zcat(n):
    vv=np.zeros(n+1)
    vv[0]=1
    ww=np.zeros(n+1)
    ww[-1]=1
    state=((vv+ww)/np.sqrt(2))
    return state


def zminuscat(n):
    vv=np.zeros(n+1)
    vv[0]=1
    ww=np.zeros(n+1)
    ww[-1]=1
    state=((vv-ww)/np.sqrt(2))
    return state/np.sqrt(np.sum(np.abs(state)**2))


def compass(n):
    state=xcat(n)+ycat(n)+zcat(n)
    return state/np.sqrt(np.sum(np.abs(state)**2))

def twin_fock_superpos(n):
    vv=np.zeros(n+1)
    vv[int(n/2)]=1
    vv=vv+ (expm(-(1j)*(np.pi/2)*sg.JYm(n))@vv) + (expm(-(1j)*(np.pi/2)*sg.JXm(n))@vv)
    return vv/np.sqrt(np.sum(np.abs(vv)**2))
    
#Dense observables

def JZdense(n):
    Z=JZm(1)
    jzfull=np.kron(Z,np.identity(2**(n-1)))
    for j in range(1,n-1):
        jzfull=jzfull+np.kron(np.kron(np.identity(2**(j)),Z),np.identity(2**(n-1-j)))
    jzfull=jzfull+np.kron(np.identity(2**(n-1)),Z)
    return jzfull
def JXdense(n):
    Z=JXm(1)
    jzfull=np.kron(Z,np.identity(2**(n-1)))
    for j in range(1,n-1):
        jzfull=jzfull+np.kron(np.kron(np.identity(2**(j)),Z),np.identity(2**(n-1-j)))
    jzfull=jzfull+np.kron(np.identity(2**(n-1)),Z)
    return jzfull
def JYdense(n):
    Z=JYm(1)
    jzfull=np.kron(Z,np.identity(2**(n-1)))
    for j in range(1,n-1):
        jzfull=jzfull+np.kron(np.kron(np.identity(2**(j)),Z),np.identity(2**(n-1-j)))
    jzfull=jzfull+np.kron(np.identity(2**(n-1)),Z)
    return jzfull
    
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
    
## Sparse implementations of spin matrices
def jx(n):
    X=JXm(1)
    X=csr_matrix(X)
    mats=[]
    for j in range(n):
        mats.append(sparse.kron(sparse.kron(sparse.identity(2**j),X),sparse.identity(2**((n-1)-j))))
    return sum(mats)

def jz(n):
    Z=JZm(1)
    Z=csr_matrix(Z)
    mats=[]
    for j in range(n):
        mats.append(sparse.kron(sparse.kron(sparse.identity(2**j),Z),sparse.identity(2**((n-1)-j))))
    return sum(mats)

def jy(n):
    X=JYm(1)
    X=csr_matrix(X)
    mats=[]
    for j in range(n):
        mats.append(sparse.kron(sparse.kron(sparse.identity(2**j),X),sparse.identity(2**((n-1)-j))))
    return sum(mats)

def jp(n):
    aa=jx(n)
    bb=jy(n)
    return (aa+((1j)*bb))
    
### Sparse implementation of range K one-axis twisting generator
def hamtk(n,k):
    Z=2*JZm(1)
    Z=csr_matrix(Z)
    #Local Z
    mats=[]
    for j in range(n):
        mats.append(sparse.kron(sparse.kron(sparse.identity(2**j),Z),sparse.identity(2**((n-1)-j))))
    
    
    ## Range k ZZ interaction
    ggg=list(np.zeros(2**n))
    ham=diags(ggg,0)
    for i in range(n):
        for l in range(1,k+1):
            ham+= mats[i]@mats[np.mod(i+l,n)]
            ham+= mats[i]@mats[np.mod(i-l,n)]
    return (1/4)*ham

