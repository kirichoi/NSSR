# -*- coding: utf-8 -*-

# COPYRIGHT 2017 Kiri Choi
# Utility for clustering and features identification/selection.

from __future__ import print_function

import numpy as np
import math
import itertools
import generate
import scipy
from sklearn import neighbors
from sklearn import cluster
from sklearn import metrics

def accuracyScore(net_mat1, net_mat2):
    """
    Accuracy classification score
    """
    
    return metrics.accuracy_score(net_mat1, net_mat2)

#@profile
def getListOfCombinations(r_ind):
    """
    Return a list of indexes for all possible combinations given the number of 
    iterables
    
    :param k_ind: list of iterables (i.e. list of indexes defined in persistant
                                     layer)
    :rtype: list
    """
    r_comb = []
    for i in range(len(r_ind)):
        r_comb.append(list(list(l) for l in list(itertools.combinations(r_ind,
                           i + 1))))
    
    r_comb = [l for subl in r_comb for l in subl]
    
    return r_comb

def getFullNumModel(p_net, num_float, num_input, num_output):
    """
    Return the total number of possible models given a persistant matrix and 
    the number of floating, input and output boundary species.
    
    :param p_net: persistant matrix
    :num_float: number of floating species
    :num_input: number of input boundary species
    :num_output: number of output boundary species
    """
    
    num_node = num_float + num_input + num_output
    s_arr = generate.generateSpeciesList(num_float, num_input, num_output)
    c_arr = generate.generateCombinationOfSpecies(s_arr)
    p_net_c = generate.generateReducedPersistanceMatrix(p_net, num_node, c_arr)
    
    if num_input > 0:
        num_none_input = len(np.where(p_net_c[num_input,:] == None)[0])
    else:
        num_none_input = 1
    if num_output > 0:
        num_none_output = len(np.where(p_net_c[:,num_input+num_float] == None)[0])
    else:
        num_none_output = 1
        
    num_none_UU = len(np.where(p_net_c[num_input:num_node,:num_node-num_output] == None)[0])
    num_none_UB = len(np.where(p_net_c[num_input:num_node,num_node:] == None)[0])
    num_none_BU = len(np.where(p_net_c[num_node:,:num_node-num_output] == None)[0])
    num_none_BB = len(np.where(p_net_c[num_node:,num_node:] == None)[0])
    
    output = 0
    
    for i in range(num_float, num_none_UU):
        output += scipy.special.comb(num_none_UU, i)
        
    for i in range(num_float, num_none_UB):
        output += scipy.special.comb(num_none_UB, i)
    for i in range(num_float, num_none_BU):
        output += scipy.special.comb(num_none_BU, i)
    for i in range(num_float, num_none_BB):
        output += scipy.special.comb(num_none_BB, i)
    
    return num_none_input*num_none_output*output

# Directionality
def corr_count(pert_class):
    pass

# Magnitude
def distance(pert_output):
    
    return np.average(pert_output)

# Scaled magnitude
def distanceScaled(pert_output):
    
    return np.divide(np.average(pert_output), pert_output)

# Magnitude vs reaction number
def distancePerRate(pert_output):
    
    return np.average(pert_output, axis=1)

# TODO: Fix issue with sets with sparce clusters
def run(sorted_k_class, com_ind):
    """
    Run clustering algorithm using K-Means
    
    :param sorted_k_class: grouped classification output
    :param com_ind:
    :rtype: list
    """
    
    labels = []
    
    for i in range(len(sorted_k_class)):
        labels_temp = []
        for j in range(len(sorted_k_class[i])):
            if len(sorted_k_class[i][j]) > 0:
                k_means = cluster.KMeans(n_clusters=np.power(2, len(com_ind)))
                k_means.fit(np.array(sorted_k_class[i][j])[:, com_ind])
                labels_temp.append(k_means.labels_)
            else:
                labels_temp.append([])
        labels.append(labels_temp)
    
    return labels

def loadNpyFile(filePath):
    """
    Load a .npy file
    
    :param filePath: path to .npy file
    """
    
    return np.load(filePath)

def maximumDisc(cluster_output):
    """
    Search for perturbations with maximum discrimitory potential. Smaller 
    values mean better discriminability.
    Nan is assigned to those that does not have any reactions or does not 
    provide any discriminability (i.e. all the classification values are the 
    same)
    
    :param cluster_output: list of cluster outputs
    :rtype: array
    """

    disc_list = np.empty([len(cluster_output), len(cluster_output[0]), 
                         len(cluster_output[0][0])])
    
    for i in range(len(cluster_output)):
        for j in range(len(cluster_output[i])):
            for k in range(len(cluster_output[i][j])):
                if len(cluster_output[i][j][k]) > 0:
                    group, counts = np.unique(cluster_output[i][j][k], 
                                              return_counts=True)
                    if len(counts) > 1:
                        # Current scoring function:
                        # Penalize the deviation between counts and smaller 
                        # number of counts
                        # f(c_N) = sigma(c) + 1/N
                        val = np.add(np.std(counts), 1/len(counts))
                    else:
                        val = np.nan
                    disc_list[i][j][k] = val
                else:
                    disc_list[i][j][k] = np.nan
                                          
    return disc_list
    
# Return indexes in network matrix with maximum discriminability with index for
# list of combination of perturbation
def pinpoint(disc_list, order=1):
    
    ordermin = np.unique(np.sort(disc_list, axis=None))[:order]
    
    r_list = []
    
    for i in range(len(ordermin)):
        r_list.append(np.array(np.where(disc_list == ordermin[i])).T)
    
    return r_list

def maximumFreq(net_mat_list, t_net_ind, nt_net_ind):
    """
    
    """
    #t_net_mat = net_mat_list[t_net_ind]
    #nt_net_mat = net_mat_list[nt_net_ind[i]]
    t_net_mean = np.mean(net_mat_list[t_net_ind], axis=0)
    #nt_net_mean.append(np.mean(net_mat_list[nt_net_ind[i]], axis=0))
    
    return t_net_mean

def maximumFreqInd(net_mat_list):
    """
    In case when network matrices are selected already to have the same 
    categorization as the true network
    """
    #t_net_mat = net_mat_list[t_net_ind]
    #nt_net_mat = net_mat_list[nt_net_ind[i]]
    t_net_mean = np.mean(net_mat_list, axis=0)
    #nt_net_mean.append(np.mean(net_mat_list[nt_net_ind[i]], axis=0))
    
    return t_net_mean

def mutualInfo(net_mat1, net_mat2):
    """
    Calculate mutual information between two network matrices
    
    """
    mi = metrics.mutual_info_score(net_mat1.flatten(), net_mat2.flatten())
    
    return mi
    
def mutualInfoAdj(net_mat1, net_mat2):
    """
    Calculate adjusted mutual information between two network matrices
    
    """
    ami = metrics.adjusted_mutual_info_score(net_mat1.flatten(), 
                                            net_mat2.flatten())
    
    return ami
    
def mutualInfoNorm(net_mat1, net_mat2):
    """
    Calculate adjusted mutual information between two network matrices
    
    """
    nmi = metrics.normalized_mutual_info_score(net_mat1.flatten(), 
                                            net_mat2.flatten())
    
    return nmi
    
def serachCommon(t_net_ind):
    """
    From indices that represent all network matrix that has a reaction defined 
    by persistant layer, pick indices for network matrix that has all reactions 
    defined by persistant layer    
    
    :param t_net_ind: target network matrix indices
    :rtype: list
    """
    
    if len(t_net_ind) > 1:
        for i in range(len(t_net_ind) - 1):
            if i == 0:
                temp = t_net_ind[i]
            temp = np.intersect1d(temp, t_net_ind[i + 1])
    else:
        temp = t_net_ind[0]
        
    return temp
    

