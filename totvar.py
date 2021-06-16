# Total variation distance for photon number detection data
import numpy as np
from sympy import *
import pandas as pd
from scipy.linalg import expm, sinm, cosm
from scipy.misc import derivative
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt
from scipy import linalg
from collections import Counter

def tvdistance(id_res,bs_res,shots,n_modes):
    from collections import Counter
    ## Get the counts for each Fock state in the measurement 
    #bitstrings_id = [tuple(i) for i in id_res.samples]
    bitstrings_id = [tuple(i) for i in id_res]
    cc_id=sorted( Counter(bitstrings_id).items() )

    #bitstrings_bs = [tuple(i) for i in bs_res.samples]
    bitstrings_bs = [tuple(i) for i in bs_res]
    cc_bs=sorted( Counter(bitstrings_bs).items() )
    ####################################################
    #Change structure of cc_id and cc_bs 
    #to the form cc_id2[j]=[photon string]+[empirical probability]

    cc_id2=[]
    for j in range(len(cc_id)):
        aa=list(cc_id[j][0])
        #Probability
        aa.append(cc_id[j][1]/shots)
        cc_id2.append(aa)
    cc_bs2=[]
    for j in range(len(cc_bs)):
        aa=list(cc_bs[j][0])
        #Probability
        aa.append(cc_bs[j][1]/shots)
        cc_bs2.append(aa)
    ###########################################################
    #Compute total variation distance
    dist=0
    #Convert the photon counts to list from np.array so that list
    #membership can be computed
    xxx=[id_res[j].tolist() for j in range(len(id_res))]
    yyy=[bs_res[j].tolist() for j in range(len(bs_res))]
    for j in range(len(cc_id2)):
        if xxx[j] in yyy:
            #If the photon string is in both experiments, scan through
            #the second experiment to find the empirical probablity
            #and calculate the distance
            for k in range(len(cc_bs2)):
                if cc_id2[j][:n_modes]==cc_bs2[k][:n_modes]:
                    dist+= (1/2)*np.abs(cc_id2[j][-1]-cc_bs2[k][-1])
        else:
            #If the photon string is not in both experiments
            dist+= (1/2)*cc_id2[j][-1]
    #Get the photon strings in bs_res.samples that are not in id_res.samples
    for k in range(len(cc_bs2)): 
        if not(yyy[k] in xxx):
            dist+=(1/2)*cc_bs2[k][-1]
    return dist

