# -*- coding: utf-8 -*-

# COPYRIGHT 2017 Kiri Choi
# Utility functions

from __future__ import print_function

import numpy as np
import pandas 

#@profile
def getNumReactions(net_mat):
    """
    Return total number of reactions from a network matrix
    
    :param net_mat: network matrix
    :rtype: int
    """
    return np.count_nonzero(net_mat)

def getPersistantIndex(p_layer):
    """
    Return indices that has values defined from persistant layer. Ignores 0 
    which means no reaction in persistant layer
    
    :param p_layer: persistant layer
    :rtype: index
    """
    
    ones = np.where(p_layer == 1)
    
    return np.array(ones).T

#@profile
def getPersistantOrder(net_mat, p_net):
    """
    If persistant layer is assigned, check the order of reactions from a 
    network matrix that corresponds to the reactions defined by persistant 
    layer
    
    :param net_mat: network matrix
    :param p_layer: persistant layer
    :rtype: list
    """
    
    num_node = np.shape(net_mat)[0]
    
    net_mat
    
    k_ind = []
    
    if p_net is not None:
        if p_net.all() is not np.array(None):
            for i in range(len(np.array(np.nonzero(p_net)).T)):
                sum_ind = ((np.array(np.nonzero(p_net)).T[i][0]*num_node) + 
                            np.array(np.nonzero(p_net)).T[i][1])
                k_ind.append(np.count_nonzero(net_mat.flatten()[:sum_ind]))
    
    return k_ind

def getUniqueNetwork(net_mat_list, num_sets, num_node):
    """
    Return unique network matrices from list of network matrices
    
    :param net_mat_list: list of network matrix
    :param num_set: number of set
    :param num_node: number of nodes
    """
    
    flat = net_mat_list.reshape(num_sets, num_node*num_node)
    
    ca = np.ascontiguousarray(flat).view(np.dtype((np.void, 
                                           flat.dtype.itemsize*flat.shape[1])))
    _, idx = np.unique(ca, return_index=True)
    
    u_mat = np.unique(ca).view(flat.dtype).reshape(-1, flat.shape[1])
    
    u_mat_reshape = u_mat.reshape(len(u_mat), num_node, num_node)
    
    # Sort both index and network matrices in ascending order
    idxarg = idx.argsort()
    
    idx = idx[idxarg]
    u_mat_reshape = u_mat_reshape[idxarg]
    
    return u_mat_reshape, idx

def groupReaction(net_mat_list):
    """
    Group network matrices by reactions present. Return index of network 
    matrices for each present reaction in num_node x num_node list
    
    :param net_mat_list: list of network matrices
    :rtype: list of index
    """
    
    num_node = np.shape(net_mat_list)[1]
    
    r_g = []
    
    for i in range(num_node):
        r_g_temp = []
        for j in range(num_node):
            r_g_temp.append(np.argwhere(net_mat_list[:,i,j] == 1).flatten())
            
        r_g.append(r_g_temp)
    
    return r_g

def isUnique(net_mat, net_mat_list):
    """
    Check whether a network matrix is already part of another array.
    
    :param net_mat: array to check
    :param u_net_mat: array to compare
    """
    
    unique = False
    
    if len(net_mat_list) == 0:
        unique = True
    else:
        unique = (np.sum(np.abs(np.array(net_mat_list) - net_mat), 
                                                        axis=(1,2)) != 0).all()
    
    return unique

#@profile
def perturbRate(rr, k_ind, pert_amt):
    """
    Perturb rate constants and returns steady state values
    
    :param rr: RoadRunner instance
    :param k_ind: list of rate constant index to perturb
    :param pert_amt: amount to perturb
    :rtype: array
    """
    
    rr.resetAll() # ALWAYS RESET
    rr.conservedMoietyAnalysis = True
    # Get string of rate constant to perturb
    rr.model.setGlobalParameterValues(k_ind, np.repeat(pert_amt, len(k_ind)))
    
    return rr.steadyStateNamedArray()
       
def perturbSpecies(rr, s_ind, pert_amt):
    """
    Perturb species concentrations and returns steady state values
    
    :param rr: RoadRunner instance
    :param k_ind: list of species index to perturb
    :param pert_amt: amount to perturb
    :rtype: array
    """
    
    rr.resetAll() # ALWAYS RESET
    rr.conservedMoietyAnalysis = True
    # Get string of species to perturb
    
    # If perturbing bounary species
    if s_ind < len(rr.getBoundarySpeciesIds()):
        rr.model.setBoundarySpeciesConcentrations(s_ind, np.repeat(pert_amt, 
                                                                   len(s_ind)))
    # If perturbing floating species
    else:
        s_ind = s_ind - len(rr.getBoundarySpeciesIds())
        rr.model.setFloatingSpeciesConcentrations(s_ind, np.repeat(pert_amt, 
                                                                   len(s_ind)))
    
    return rr.steadyStateNamedArray()

def getDataFrame(arr):
    """
    Convenience method to print out wide 2D matrix using pandas
    
    :param arr: 2D array
    """
    
    pandas.set_option('display.width', 500)
    pd = pandas.DataFrame(arr)
    
    return pd

def resort(arr, species_list):
    """
    Sort an array according to species order
    
    :param arr: array to sort
    :param species_list: list of species
    :rtype: array
    """
    
    return arr[:,np.array(species_list).argsort()]

   
    