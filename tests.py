import numpy as np
from numpy import linalg
import scipy as sp
from scipy.special import factorial
from scipy import sparse
from scipy.sparse import csr_matrix,coo_matrix
%matplotlib inline
import matplotlib as mpl
from matplotlib import pyplot as plt

import fragmented as frag
from fragmented import Arbfrag
import pair_fragmented as pfrag
from pair_fragmented import Pairfrag

#Number of particles
N=64


# #Generate state instance
# test=frag.Arbfrag(N)
# test.N
# teststate=test.state(32,-np.pi/4,-np.pi/6)
# #Check the basis indices
# test.g()
#
#
# #Plot the state
# plt.figure(figsize=(8,4))
# axes = plt.gca()
# axes.plot(teststate,'k-')
# np.sum(np.square(teststate))

#Number of particles
N=64
#Generate state instance
test=pfrag.Pairfrag(N)
state=test.state(10,-np.pi/4,np.pi/3,np.pi/4,np.pi/4)
#Plot the state
plt.figure(figsize=(8,4))
axes = plt.gca()
axes.plot(state,'k-')
