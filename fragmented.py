import numpy as np
from numpy import linalg
import scipy as sp
from scipy.special import factorial
from scipy import sparse
from scipy.sparse import csr_matrix,coo_matrix


### Works for two fragments. Call Arbfrag(N) to initialize. 
### The state has the form (a_{1}^{*} + \tan \theta_{1} a_{2}^{*})^{N-m}(a_{1}^{*} + \tan \theta_{2} a_{2}^{*})^{m} \ket{0,0}
### with m\le N/2

class Arbfrag:
    def __init__(self,numPtcl):
        self.N=numPtcl
        
    def g(self):
        N=self.N
        g=np.zeros((N+1,N+1))
        c=0
        for l in range(N+1):
            for m in range(N+1):
                g[l,m]=c
                c+=1
        return g
    
    ### A is a sparse matrix that defines the fragments.
    
    def A(self,theta):
        N=self.N
        g=self.g()
        arow=[]
        acol=[]
        adata=[]
        for k in range(N+1):
            for j in range(N):
                    acol.append(g[j,k])
                    arow.append(g[j+1,k])
                    adata.append(np.sqrt(j+1))

        brow=[]
        bcol=[]
        bdata=[]
        for k in range(N+1):
            for j in range(N):
                    bcol.append(g[k,j])
                    brow.append(g[k,j+1])
                    bdata.append(np.sqrt(j+1))

        arow=np.int_(np.asarray(arow))
        acol=np.int_(np.asarray(acol) )
        brow=np.int_(np.asarray(brow))
        bcol=np.int_(np.asarray(bcol) )
        adata=np.asarray(adata)
        bdata=np.asarray(bdata)
        AA=coo_matrix((adata, (arow, acol)), shape=((N+1)**2, (N+1)**2)).tocsr()
        BB=coo_matrix((bdata, (brow, bcol)), shape=((N+1)**2, (N+1)**2)).tocsr()
        #Taking the matrix power via ** gives a reliable state only for small N (N\lesssim 100). However, one can take a small power to reduce the
        #number of matrix multiplications during state generation.
        #H=(AA-(np.tan(theta)*BB))**5
        H=(AA-(np.tan(theta)*BB))
        return H
    
    def state(self,m,theta1,theta2):
        #Need to demand that fragments are specified by increasng number of particles.
        #This allows to implement a prophylactic if destructive interference is expected between fragments
        # which alternates the matrix multiplications to avoid errors.
        N=self.N
        g=self.g()
        if m>N/2:
            return print('m must be greater than or equal to N/2')
        else:

            ms=np.zeros((N+1)**2)
            ms[0]=1

            xx=ms
            for j in range(m):
                xx=( self.A(theta2).dot(xx) )/np.linalg.norm(xx)
                xx=( self.A(theta1).dot(xx) )/np.linalg.norm(xx)
            for j in range(np.int(N-(2*m))):
                xx=( self.A(theta1).dot(xx) )/np.linalg.norm(xx)

            state=[]
            for k in range(N+1):
                for j in range(N+1):
                    if j+k==N:
                        state.append(xx[np.int(g[j,k])])
            state=state/np.linalg.norm(state)
            return state
