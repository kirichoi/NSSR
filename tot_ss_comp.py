# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 15:23:16 2017

@author: Kiri
"""

import numpy as np
import scipy

def numSolutions(N, fN, bN=0):
    """
    Computes total number of possible solutions given the number of 
    total/floating/boundary species.
    
    :param N: Number of total species
    :param fN: Number of floating species
    :param bN: Number of boundary species
    """
    rank_j = scipy.misc.comb(N, 2)
    
    q = scipy.misc.factorial(rank_j/2)
    
    total = 2*np.square(q)
    
    return total


