# -*- coding: utf-8 -*-

# COPYRIGHT 2017 Kiri Choi
# Truth network model analysis

from __future__ import print_function

import numpy as np
import tellurium as te
import antimony
import generate
import util
import clustering

def classify(setup, s_arr, c_arr):
    """
    Ground truth classification. Returns initial perturbation response, 
    perturbation response, classification, and reaction index
    
    :param g_truth: ground truth network matrix
    :param s_truth: ground truth species concentrations
    :param k_truth: ground truth rate constants
    :param num_node: ground truth numbder of nodes
    :param num_bound: ground truth numbder of boundary species
    :param k_pert: perturbation amount
    :param Thres: classification threshold
    :rtype: list
    """
    antimony.clearPreviousLoads()
    
    # Strip and translate to string
    t_s = setup.t_s.astype('str')
    t_k = setup.t_k[setup.t_k != np.array(0)].astype('str')
    
    #t_k_count = np.count_nonzero(setup.t_net)
    t_ant = generate.generateAntimonyNew(setup.t_net, t_s, t_k, s_arr, c_arr)
    
    #r_ind = np.array(np.where(setup.t_net != np.array(0))).T
    r_ind = util.getPersistantOrder(setup.t_net, setup.p_net)
    rr = te.loada(t_ant)
    rr.reset() # ALWAYS RESET
    rr.conservedMoietyAnalysis = True
    pert_i = rr.steadyStateNamedArray() # Initial steady state
    
    r_comb = clustering.getListOfCombinations(r_ind)
    
    # Pertubation for rate constants
    k_pert_output_i = np.empty([len(r_comb), setup.num_float])

    for i in range(len(r_comb)):
        k_pert_output_i[i] = util.perturbRate(rr, r_comb[i], setup.k_pert)

    # Classification for rate constants
    k_class_i = np.empty([len(r_comb), setup.num_float], dtype=int)
    
    for i in range(len(r_comb)):
        for j in range(setup.num_float):
            k_diff = (k_pert_output_i[i][j] - pert_i[0][j])
            if (np.abs(k_diff) > setup.Thres):
                if k_diff < 0.:
                    k_class_i[i][j] = 1
                else:
                    k_class_i[i][j] = 2
            else:
                k_class_i[i][j] = 0    
    
    antimony.clearPreviousLoads()
    
    return pert_i[0], k_pert_output_i, k_class_i

def compareClass(t_analysis, k_class):
    """
    Return indices of network matrices that fall into the same category and 
    those that does not fall into the same category as the result from true 
    network 
    
    :param t_analysis:
    :param k_class:
    """

    t_net_ind = []
    nt_net_ind = []
    
    for i in range(len(k_class)):
        if np.array_equal(t_analysis[2], k_class[i]):
            t_net_ind.append(i)
        else:
            nt_net_ind.append(i)
            
    return t_net_ind, nt_net_ind

#@profile
def compareIndClass(t_analysis, k_class_i):
    """
    Checks a single instance of classification against the true result. Returns 
    True if classification is identical and false otherwise
    
    :param t_analysis:
    :param k_class:
    """
    partial = False

    if np.array_equal(t_analysis, k_class_i):
        partial = True
            
    return partial



#def compareClass(p_r_ind, t_analysis, k_class, net_ind_group):
#    """
#    Return indices for network matrices that fall into the same category and 
#    those that does not fall into the same category as the output of ground 
#    truth model
#    
#    :param p_r_ind: persistant index
#    :param t_analysis: output of ground truth classification
#    :param k_class: classification output resulting from perturbing reaction
#    :param net_ind_group: grouped reaction index
#    :rtype: list
#    """
#    
#    t_net_ind = []
#    nt_net_ind = []
#    
#    for i in range(len(p_r_ind)):
#        row = p_r_ind[i][0]
#        col = p_r_ind[i][1]
#        
#        # Get generated classification from target indices
#        t_k_class = sorted_k_class[row][col]
#        
#        # Get truth classification from target indices
#        comp1 = np.array([np.in1d(t_analysis[3].T[0], row), 
#                         np.in1d(t_analysis[3].T[1], col)])
#
#        truth_k_class = t_analysis[2][comp1.all(axis=0)]
#
#        # Get indices where generated classification and truth 
#        # classification is the same 
#        # TODO: Currently this matches all binary values
#        ind_id = np.where((t_k_class == truth_k_class).all(axis=1))[0]
#        
#        # Network matrix indices that match with truth classification
#        t_net_ind_i = net_ind_group[row][col][ind_id]
#        # Network matrix indices that does not match with truth classification
#        nt_net_ind_i = np.setdiff1d(net_ind_group[row][col], t_net_ind_i)
#        t_net_ind.append(t_net_ind_i)
#        nt_net_ind.append(nt_net_ind_i)
#    
#    return t_net_ind, nt_net_ind

