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

#Two mode systems

def fastqfiloss_fullrank(ang,n,k):
    #Compute noisy case with usual formula for QFI but using a full rank state
    rr=dicketrace(n,k,int(n/2))
    size=n-k
    small=(10**-10)
    dd=((1-small)*rr)+ (small*(1/(n-k+1))*np.eye(int(n-k+1)))
    specsy=np.diag(dd)
    #Rotate the eigenvectors
    yy=sg.JYm_sparse(size)
    rotveclist=[]
    for j in range(len(specsy)):
        vv=np.zeros(size+1).astype(np.complex128)
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
        fishy+= 2*ffy*(np.abs(ggy)**2)
    for i in range(size+1):
        for j in range(i+1,size+1):
            #Off-diagonal contributions
            ffy=(1/(specsy[i]+specsy[j]))
            ggy=aaa[i,j]
            fishy+= 2*2*ffy*(np.abs(ggy)**2)
    
    return (1/n)*np.real(fishy)
    
#Four mode systems
from itertools import*
# C XXXII p.126 for discussion of indices for
# four mose bosonic system with fixed particle number

def indexpairs(n,modes):
    f=0
    gg=[]
    for k in product(range(n+1),repeat=modes):
        if sum(k)==n:
            f+=1
            gg.append([f-1,list(k)])
    return gg
    
def indexpairsdict(n,modes):
    f=0
    fff={}
    for k in product(range(n+1),repeat=modes):
        if sum(k)==n:
            f+=1
            fff[str(list(k))]=f-1
    return fff
    
def jplus_ab_fourmode_sparse(n,modes):
    #modes=4 for the code below to work
    f=0
    yrow=[]
    ycol=[]
    ydata=[]
    ccc=indexpairsdict(n,modes)
    hhh=indexpairs(n,modes)
    for j in range(len(hhh)):
        #For Dicke m_0,m_1,m_2,m_3, we create m_0+1,m_1-1,m_2,m_3, put their indices
        #to the index lists for the sparse matrix, 
        #and add the matrix element of J_{+} on first two modes.
        if (hhh[j][1][0]!=n)&(hhh[j][1][1]!=0):
            xx=int(ccc[str(hhh[j][1])])
            ycol=ycol+[xx]
            ppp=np.array([1,0,0,0])+(np.array(hhh[j][1])-np.array([0,1,0,0]))
            yrow=yrow+[int(ccc[str(list(ppp))])]
            ydata.append(np.sqrt((hhh[j][1][0]+1)*hhh[j][1][1]))
    size=int(comb(n+modes-1,modes-1))
    yyy=coo_matrix((ydata, (yrow, ycol)), shape=(size,size)).tocsr()
    return yyy

def jplus_cd_fourmode_sparse(n,modes):
    #modes=4 for the code below to work
    f=0
    yrow=[]
    ycol=[]
    ydata=[]
    ccc=indexpairsdict(n,modes)
    hhh=indexpairs(n,modes)
    for j in range(len(hhh)):
        #For Dicke m_0,m_1,m_2,m_3, we create m_0-1,m_1,m_2+1,m_3, put their indices
        #to the index lists for the sparse matrix, 
        #and add the matrix element of J_{+} on last two modes.
        if (hhh[j][1][2]!=n)&(hhh[j][1][3]!=0):
            xx=int(ccc[str(hhh[j][1])])
            ycol=ycol+[xx]
            ppp=np.array([0,0,1,0])+(np.array(hhh[j][1])-np.array([0,0,0,1]))
            yrow=yrow+[int(ccc[str(list(ppp))])]
            ydata.append(np.sqrt((hhh[j][1][2]+1)*hhh[j][1][3]))
    size=int(comb(n+modes-1,modes-1))
    yyy=coo_matrix((ydata, (yrow, ycol)), shape=(size,size)).tocsr()
    return yyy

def jy_ab(n,modes):
    aa=jplus_ab_fourmode_sparse(n,modes)
    bb=aa.conj().T
    return (aa-bb)/(2*(1j))

def jy_cd(n,modes):
    aa=jplus_cd_fourmode_sparse(n,modes)
    bb=aa.conj().T
    return (aa-bb)/(2*(1j))

def jz_ab_fourmode_sparse(n,modes):
    #modes=4 for the code below to work
    f=0
    yrow=[]
    ycol=[]
    ydata=[]
    ccc=indexpairsdict(n,modes)
    hhh=indexpairs(n,modes)
    for j in range(len(hhh)):
        xx=int(ccc[str(hhh[j][1])])
        ycol=ycol+[xx]
        yrow=yrow+[xx]
        ydata.append( (1/2)*(hhh[j][1][0] - hhh[j][1][1]) )
    size=int(comb(n+modes-1,modes-1))
    yyy=coo_matrix((ydata, (yrow, ycol)), shape=(size,size)).tocsr()
    return yyy

def jz_cd_fourmode_sparse(n,modes):
    #modes=4 for the code below to work
    f=0
    yrow=[]
    ycol=[]
    ydata=[]
    ccc=indexpairsdict(n,modes)
    hhh=indexpairs(n,modes)
    for j in range(len(hhh)):
        xx=int(ccc[str(hhh[j][1])])
        ycol=ycol+[xx]
        yrow=yrow+[xx]
        ydata.append( (1/2)*(hhh[j][1][2] - hhh[j][1][3]) )
    size=int(comb(n+modes-1,modes-1))
    yyy=coo_matrix((ydata, (yrow, ycol)), shape=(size,size)).tocsr()
    return yyy
    
#C XXXII p.124 for discussion of these indices.
def index(m,x):
    aa=x[0]*(m[1]+1)*(m[2]+1)*(m[3]+1)
    aa=aa+(x[1]*(m[2]+1)*(m[3]+1))
    aa=aa+(x[2]*(m[3]+1))
    aa=aa+x[3]
    return int(aa)

def index_to_dicke(m,b):
    v1=(m[1]+1)*(m[2]+1)*(m[3]+1)
    v2=(m[2]+1)*(m[3]+1)
    v3=(m[3]+1)
    a0=int(b/v1)
    a1=int((b-(a0*v1))/v2)
    a2=int((b-(a0*v1)-(a1*v2))/v3)
    a3=b-(a0*v1)-(a1*v2)-(a2*v3)
    return np.array([int(a0),int(a1),int(a2),int(a3)])

def double_index_to_dicke(mm):
    #mm is list of integers of length 2
    nn=mm[0]+mm[1]
    uu=np.zeros(nn+1).astype(np.complex128)
    uu[mm[1]]=1
    return uu

def qtens(n,k,mm):
    #CXXXII p.124-125
    v=[ np.array([1,0,0,0]),np.array([0,1,0,0]),np.array([0,0,1,0]),np.array([0,0,0,1])]
    Hcol=[]
    Hrow=[]
    Hdata=[]
    #Dicke branching from loss
    v=[ np.array([1,0,0,0]),np.array([0,1,0,0]),np.array([0,0,1,0]),np.array([0,0,0,1])]
    losslist=[[mm]]
    for s in range(1,k+1):
        losslist.append([losslist[s-1][r]-v[h] for r in range(len(losslist[s-1])) for h in range(4)])
    #Remove duplicates for column labels
    yy=[]
    for s in range(len(losslist)):
        yy.append([])
        for j in losslist[s]:
            if list(j) not in yy[s]:
                yy[s].append(list(j))
                
    for s in range(k):
        Hcol=Hcol+[index(mm,a) for a in yy[s] for h in range(4)]
        Hrow=Hrow+[index(mm,np.array(a)-v[h]) for a in yy[s] for h in range(4)]
        Hdata=Hdata+[c[h]/np.sum(c) for c in yy[s] for h in range(4)]
    Hrow=np.int_(np.asarray(Hrow))
    Hcol=np.int_(np.asarray(Hcol) )
    Hdata=np.asarray(Hdata)
    H=coo_matrix((Hdata, (Hrow, Hcol)), shape=(int(np.prod(mm+1)),int(np.prod(mm+1)))).tocsr()
    return H

def fastqfiloss_fullrank(n,k,aaa):
    #jyab and jybc should be local J_{y} for n-k particles.
    #n should be 4*(n/4) and mm=np.array([n/4,n/4,n/4,n/4])
    mm=np.array([n/4,n/4,n/4,n/4])
    qq=qtens(n,k,mm)
    pp=np.zeros(int((mm[0]+1)**4)).astype(np.complex128)
    pp[-1]=1
    ff=pp
    for j in range(k):
        ff=qq@ff
    #Get the nonzero elements
    uu=np.nonzero(ff)[0]
    dickelist=[index_to_dicke(mm,uu[j]) for j in range(len(uu))]
    weights=[ff[uu[j]] for j in range(len(uu))]
    #Set up the states for local interferometry. Need to go to
    #basis with n_{K} total particles. Each Dicke state has this total
    #number
    #ttt=indexpairs(64-2,4)
    ccc=indexpairsdict(n-k,4)
    nk=n-k
    small=(10**-10)
    srow=[]
    scol=[]
    sdata=[]
    size=int(comb(n-k+4-1,4-1))
    for j in range(len(uu)):
        srow=srow+[ ccc[str(list(dickelist[j]))] ]
        scol=scol+[ ccc[str(list(dickelist[j]))] ]
        sdata=sdata+[ ( (1-small)*weights[j] ) ]
    dd=coo_matrix((sdata, (srow, scol)), shape=(size,size)).tocsr()
    dd=dd+((small/size)*identity(size))

    specsy=dd.diagonal()
    #Rotate the eigenvectors. Although in the present case, the QFI
    #is independent of \theta, so just get the eigenvectors
    #rotveclist=[identity(size).getcol(j) for j in range(size)]

#     fishy=0
#     #Instead of using the rotated vectors, just multiply
#     #delrhoy by sparse matrices, then access matrix elements.
#     delrho1y=-(1j)*( aaa@las.expm_multiply( -(1j)*ang*aaa, las.expm_multiply(-(1j)*ang*aaa,dd).conj().T))
#     delrho2y=(1j)*las.expm_multiply( -(1j)*ang*aaa, las.expm_multiply(-(1j)*ang*aaa,dd).conj().T)@aaa
#     delrhoy=delrho1y+delrho2y
#     ff=las.expm_multiply((1j)*ang*aaa,las.expm_multiply((1j)*ang*aaa,delrhoy).conj().T)

#     #No rotation at all
#     delrho1y=-(1j)*(aaa@dd)
#     delrho2y=(1j)*(dd@aaa)
#     delrhoy=delrho1y+delrho2y
#     ww=find(delrhoy)
#     for i in range(len(ww[0])):
#         for j in range(len(ww[1])):
#             if (specsy[ww[0][i]]>(10**(-8)))and(specsy[ww[1][j]]>(10**(-8))):
#                 ffy=(1/(specsy[ww[0][i]]+specsy[ww[1][j]]))
#                 ggy=delrhoy[ww[0][i],ww[1][j]]
#                 fishy+= 2*ffy*(np.abs(ggy)**2)

    #Sparse SLD method
    rr=find(aaa)
    sldrows=list(rr[0])
    sldcols=list(rr[1])
    jydata=list(rr[2])
    slddata=[]
    for j in range(len(sldrows)):
        bbb=(2*(1j))*jydata[j]*(specsy[sldcols[j]]-specsy[sldrows[j]])/(specsy[sldcols[j]]+specsy[sldrows[j]])
        slddata.append(bbb)
    sld=coo_matrix((slddata, (sldrows, sldcols)), shape=(size,size)).tocsr()
    pro=dd@sld@sld
    #return (2/n)*np.real(fishy)
    return (2/n)*np.real(pro.trace())

