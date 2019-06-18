# -*- coding: utf-8 -*-

# COPYRIGHT 2017 Kiri Choi
# Generate random network matrix with randomized initial species concentrations
# and rate constants. Convert network matrix to Antimony string.

from __future__ import print_function

import numpy as np
import os
import itertools

#@profile
def generateAntimony(net_mat, s_init, k_init, s_arr, c_arr):
    """
    Generate Antimony string given network matrix and initial values.
    
    :param net_mat: network matrix
    :param s_init: initialized species value array
    :param k_init: initialized rate constants value array
    :param s_arr: string list of species
    :param c_arr: matrix of species combination
    :rtype: str
    """
    
    ant_str = str()
    
    # Species definition - fix issue with unordered species in output
    ant_str = "species "
    ant_str += ', '.join(s_arr)
    ant_str += (";\n\n")

    # Identify number of rate constants
    rc_list = []
    
#    num_rc = len(np.where(abs(net_mat) == 1)[0])
#    for i in range(num_rc):
#        rev_list.append("k" + str(i + 1))
    
    # Reactions
    j_ind = 0
    k_ind = 0
    for i in range(len(net_mat)):
        for j in range(len(net_mat[i])):
            # Reaction exists
            if net_mat[i][j] == 1 or net_mat[i][j] == 2:
                r_str = str()
                
                rct_s = c_arr[i]
                prd_s = c_arr[j]
                # TODO: account for activation/inhibition
                if net_mat[i][j] == 1:
                    if type(rct_s) == str:
                        rct = rct_s
                        kinetics = "k" + str(i) + "_" + str(j) + "*" + rct_s
                    else:
                        rct = ' + '.join(rct_s)
                        kinetics = "k" + str(i) + "_" + str(j) + "*" + '*'.join(rct_s)
                    
                    if type(prd_s) == str:
                        prd = prd_s
                    else:
                        prd = ' + '.join(prd_s)
                    
                # Inhibition
                elif net_mat[i][j] == 2:
                    comm_rcts = list(set(c_arr[i]).intersection(c_arr[j]))
                    uncomm_rcts = list(c_arr[i])
                    uncomm_rcts.remove(comm_rcts[0])
                    rct = ' + '.join(rct_s)
                    
                    kinetics = "k" + str(i) + "_" + str(j) + "*" + uncomm_rcts[0] + "/(" + comm_rcts[0] + " + 1)"
                    
                    uncomm_prds = list(set(prd_s) - set(comm_rcts))
                    prd = ' + '.join(prd_s)
                
                r_str += ("J" + str(j_ind) + ": " + rct + " -> "
                                    +  prd + "; " + kinetics + ";")
                j_ind += 1
                
                rc_list.append("k" + str(i) + "_" + str(j) + " = " + str(k_init[k_ind]) + ";")
                k_ind += 1
#                        rc_ind += 1
#                        # Forward rates
#                        for j in np.where(net_mat[:,i] == 1)[0]:
#                            forward += ("*" + s_arr[j])
#                            # Forward activation
#                            for k in np.where(net_mat[:,i] == 2)[0]:
#                                forward += ("*(" + s_arr[k] + "/(1 + " + s_arr[k] + "))")
#                            # Forward inhibition
#                            for k in np.where(net_mat[:,i] == -2)[0]:
#                                forward += ("*(1/(1 + " + s_arr[k] + "))")
#                        # Backward rates
#                        if len(np.where(net_mat[:,i] == -1)[0]) > 0:
#                            for l in np.where(net_mat[:,i] == 1)[0]:
#                                forward += (" - " + str(rev_list[rc_ind]) + "*" + s_arr[l])
#                                # Forward activation
#                                for m in np.where(net_mat[:,i] == 2)[0]:
#                                    forward += ("*(" + s_arr[k] + "/(1 + " + s_arr[m] + "))")
#                                # Forward inhibition
#                                for m in np.where(net_mat[:,i] == -2)[0]:
#                                    forward += ("*(1/(1 + " + s_arr[m] + "))")
#                            rc_ind += 1
#                        forward += (";\n")
                                            
                ant_str += r_str  + '\n'
    
    ant_str += '\n'
    
    # Species initialization
    for i in range(len(s_init)):
        ant_str += (s_arr[i] + ' = ' + str(s_init[i]) + '; ')
        
    ant_str += '\n'
    
    # Rate constants
    ant_str += ' '.join(rc_list)
    
    return ant_str
    
#def generateAntimony(net_mat, s_init, k_init, num_node, num_bound):
#    """
#    Generate Antimony string given network matrix and initial values
#    
#    :param net_mat: network matrix
#    :param s_init: initialized species value
#    :param k_init: initialized rate constants value
#    :param num_node: number of nodes
#    :param num_bound: number of boundary species
#    :rtype: str
#    """
#    
#    ant_str = str()
#    
#    # Species definition - fix issue with unordered species in output
#    ant_str = "species "
#    
#    s_arr = np.empty(num_node).astype('str')
#    bound_ind = np.arange(num_bound)
#    
#    for i in range(num_node): # Generate string array of all species
#        if i in bound_ind: # Boundary species
#            s_arr[i] =  ("$S" + str(i))
#        else: # Floating species
#            s_arr[i] = ("S" + str(i))
#            
#    ant_str += ', '.join(s_arr)
#    ant_str += (";\n")
#    
#    # Identify reversible reactions and keep track of rate constants involved
#    # in the reaction
#    rev_list = np.zeros((num_node, num_node), dtype='int')
#    k_ind = 0 # Number of rate constants
#    for i in range(num_node):
#        for j in range(num_node):
#            if net_mat[i][j] == 1:
#                rev_list[i][j] = k_ind
#                k_ind += 1
#    
#    # Reactions
#    j_ind = 0 # Number of reactions
#    for i in range(num_node):
#        for j in range(num_node):
#            if net_mat[i][j] == 1:
#                if net_mat[j][i] == 1: # Reversible reaction
#                    if j < i : # Ignore duplicate in reversiable reactions
#                        pass
#                    else: 
#                        ant_str += ("J" + str(j_ind) + ": " + s_arr[i] + " -> "
#                        + s_arr[j] + "; k" + str(rev_list[i][j]) + "*" + 
#                        s_arr[i] + " - k" + str(rev_list[j][i]) + "*" + 
#                        s_arr[j] + "\n")
#                        j_ind += 1
#                else: # Non-reversible reaction
#                    ant_str += ("J" + str(j_ind) + ": " + s_arr[i] + " -> "  
#                    + s_arr[j] + "; k" + str(rev_list[i][j]) + "*" + s_arr[i] +
#                    "\n")
#                    j_ind += 1
#                        
#    # Species initialization
#    for i in range(num_node):
#        ant_str += (s_arr[i] + ' = ' + s_init[i] + '; ')
#        
#    ant_str += '\n'
#    
#    # Rate constants
#    for i in range(k_ind):
#        ant_str += ('k' + str(i) + ' = ' + k_init[i] + '; ')
#    
#    return ant_str

def generateCombinationOfSpecies(s_arr):
    """
    Generate a matrix with combinations of species.
    
    :param s_arr: string list of species
    """
    #TODO: Add null
    #c_s = s_arr.tolist() + list(itertools.combinations(s_arr, 2))
    c_s = s_arr.tolist() + list(itertools.combinations_with_replacement(s_arr, 2))
    
    return c_s

def generateEmptyNetwork(c_arr, fval = None):
    """
    Generate a matrix with None values with given size.
    
    :param c_arr: matrix of species combination
    """
    if type(c_arr) == int:
        num_node = c_arr
    else:
        num_node = len(c_arr)
    
    net_mat = []
    
    for i in range(num_node):
        net_mat_t = []
        for j in range(num_node):
            net_mat_t.append(fval)
        net_mat.append(net_mat_t)            
        
    return np.array(net_mat)

#@profile
def generateInitialSpecies(num_node, s_max, p_s=None):
    """
    Generate randomized initial values for species in float16 and to string.
    Optionally, it is possible to set persistant layer for initial species
    to generate randomized values while keeping assigned values constant.
    
    :param num_node: number of nodes
    :param s_max: maximum species concentration
    :param p_s_layer: species persistant layer
    :rtype: array
    """
    
    # TODO: Set minimum?
    s_init = np.random.uniform(0.01, s_max, 
                               num_node).round(6) # 6 decimals
    if p_s is not None:
        if p_s.all() is not np.array(None):
            p_s_ind = np.where(p_s != np.array(None))
            s_init[p_s_ind] = p_s[p_s != np.array(None)]
    
    return s_init.astype('str')

#@profile
def generateInitialRateConstants(num_r, k_max, r_ind=None, p_k=None):
    """
    Generate randomized initial values for parameters in float16 and to string.
    Optionally, it is possible to set persistant layer for rate constants to 
    generate randomized values while keeping assigned values constant.
    
    :param k_count: number of reactions
    :param k_max: maximum rate constant
    :param k_ind: order of persistant reaction
    :param p_k: rate constant persistant layer
    :rtype: array
    """
    
    # TODO: Set minimum?
    k_init = np.random.uniform(0.01, k_max, 
                               num_r).round(6) # 6 decimals
    if p_k is not None:
        if p_k.all() is not np.array(None):
            r_ind = np.array(r_ind)
            k_init[r_ind] = p_k.flatten()[p_k.flatten() != 
                                                np.array(None)]
    
    return k_init.astype('str')

def generateRandomNetwork(s_arr, c_arr, s_ind, p_net):
    """
    Generate randomized network matrix
    
    :param s_arr: array of species
    :param c_arr: array of species combination
    :param s_ind: 
    :param num_input: number of boundary input
    :param num_output: number of boundary output
    :param p_net: persistant layer
    :rtype: array
    """

    # Number of combinations
    num_comb = len(c_arr)
    
    # Minimum number of reactions
    r_s_ind_p = []

    net_mat = np.zeros((num_comb, num_comb), dtype=int)
    
    # Incorporate persistance layer
    if len(np.where(p_net == 1)) > 0:
        p_inda = np.where(p_net == 1)
        net_mat[p_inda[0], p_inda[1]] = 1
        p_indi = np.where(p_net == 2)
        net_mat[p_indi[0], p_indi[1]] = 2
    
    # Random generation
    for i in range(num_comb):
        # Pick available spaces
        unk = np.where(p_net[i] == None)[0]
        unk_num = len(unk)
        if unk_num > 0:
            # Assign random reactions
            rand_val = np.random.choice(2, size=unk_num, p=[0.99, 0.01])
            
            # Inhibition must be specific to bi-bi
            if i >= len(s_arr):
                ti = unk[np.where(unk > len(s_arr))[0]]
                ti_i = np.where(unk > len(s_arr))[0]
                ti_f = []
                # Make sure row and column tuples has one and only one common species
                for k in range(len(ti)):
                    if c_arr[i][0] not in c_arr[ti[k]] and c_arr[i][1] not in c_arr[ti[k]]:
                        pass
                    else:
                        ti_f.append(ti_i[k])
                ti_f = np.array(ti_f)
                if len(ti_f) > 0:
                    bi_rand = np.random.choice([0, 2], size=len(ti_f), p=[0.999, 0.001])
                    rand_val[ti_f] = bi_rand
            net_mat[i][unk] = rand_val
        # Log index of species involved in reactions
        if len(np.where(net_mat[i] != 0)[0]) > 0:
            r_s_ind_p.append([i])
            r_s_ind_p.append(np.where(net_mat[i] != 0)[0].astype(int))

    r_s_ind_p = [item for sublist in r_s_ind_p for item in sublist]

    # Make sure each species are involved in at least one reaction
    for i in range(len(s_ind)):
        # Species already involved in reactions
        if [j for j in r_s_ind_p if j in s_ind[i]]:
            pass
        else:
            unk = []
            unk1 = []
            unk2 = []
            # Boundary input stays as input
            if 'I' in s_arr[i]:
                while len(unk) == 0:
                    i1 = np.random.choice(s_ind[i])
                    unk = np.where(p_net[i1,:] == None)[0].astype(int)
                i2 = np.random.choice(unk)
                net_mat[i1][i2] = 1
            # Boundary output stays as output
            elif 'X' in s_arr[i]:
                while len(unk) == 0:
                    i1 = np.random.choice(s_ind[i])
                    unk = np.where(p_net[:,i1] == None)[0].astype(int)
                i2 = np.random.choice(unk)
                net_mat[i2][i1] = 1
            # Floating species
            else:
                while len(unk1) == 0:
                    i11 = np.random.choice(s_ind[i])
                    unk1 = np.where(p_net[i11,:] == None)[0].astype(int)
                while len(unk2) == 0:
                    i12 = np.random.choice(s_ind[i])
                    unk2 = np.where(p_net[:,i12] == None)[0].astype(int)
                i21 = np.random.choice(unk1)
                i22 = np.random.choice(unk2)
                if np.random.rand() < 0.5:
                    net_mat[i11][i21] = 1
                else:
                    net_mat[i22][i12] = 1
        
    return net_mat

#@profile
#def generateRandomNetwork(num_node, num_input, num_output, p_layer=None):
#    """
#    Generate randomized network matrix
#    
#    :param num_node: number of nodes
#    :param num_input: number of boundary inputs
#    :param num_output: number of boundary outpus
#    :param p_layer: persistant layer
#    :rtype: array
#    """
#    
#    num_bound = num_input + num_output
#    
#    # Persistant layer flag
#    p_flag = False
#    
#    # Persistant layer check
#    if p_layer is not None:
#        if p_layer.all() is not np.array(3):
#            p_flag = True
#    
#    # Generate random matrix, assign boundary and floating species
#    input_ind = np.arange(num_input)
#    output_ind = np.arange(num_input, num_bound)
#    
#    #net_mat = np.random.randint(2, size=(num_node, num_node))
#    net_mat = np.random.choice(2, size=(num_node, num_node), p=[0.8, 0.2])
#
#    # Update network matrix according to persistant layer if it exists
#    # Throughout the process, reset the network matrix to keep the persistant
#    # layer
#    if p_flag:
#        p_l_ind = np.where(p_layer != np.array(3))
#        p_l_val = p_layer[p_layer != np.array(3)]
#        net_mat[p_l_ind] = p_l_val
#        
#    # No auto-regulation
#    np.fill_diagonal(net_mat, 0)
#    if p_flag:
#        net_mat[p_l_ind] = p_l_val
#    
## Reversible reactions allowed-------------------------------------------------    
##    for i in range(num_node):
##        for j in range(num_node):
##            if net_mat[i][j] == 1 and net_mat[j][i] == 1:
##                if p_flag:
##                    # Keep output of persistant layer unchanged
##                    comp_arr1 = np.equal(np.array([i,j]), 
##                                             np.array(p_l_ind).T).all(axis=1)
##                    comp_arr2 = np.equal(np.array([j,i]), 
##                                             np.array(p_l_ind).T).all(axis=1)
##                    if comp_arr1.any():
##                        if p_l_val[comp_arr1][0] == 1:
##                            net_mat[np.array(p_l_ind).T[comp_arr1][0][1]][
##                            np.array(p_l_ind).T[comp_arr1][0][0]] = 0
##                        else:
##                            net_mat[np.array(p_l_ind).T[comp_arr1][0][1]][
##                            np.array(p_l_ind).T[comp_arr1][0][0]] = 1
##                    elif comp_arr2.any():
##                        if p_l_val[comp_arr2][0] == 1:
##                            net_mat[np.array(p_l_ind).T[comp_arr2][0][1]][
##                            np.array(p_l_ind).T[comp_arr2][0][0]] = 0
##                        else:
##                            net_mat[np.array(p_l_ind).T[comp_arr2][0][1]][
##                            np.array(p_l_ind).T[comp_arr2][0][0]] = 1
##                    else: # No matching element in persistant layer
##                        if np.random.randint(2) == 0:
##                            net_mat[i][j] = 0
##                        else:
##                            net_mat[j][i] = 0
##                else: # No persistant layer assigned
##                    if np.random.randint(2) == 0:
##                        net_mat[i][j] = 0
##                    else:
##                        net_mat[j][i] = 0
## -----------------------------------------------------------------------------
#    
#    # No direct interaction between boundary species
#    net_mat[:num_bound,:num_bound] = 0
#    if p_flag:
#        net_mat[p_l_ind] = p_l_val
#        
#    # Make sure boundary species stay as input or output
#    net_mat[:,input_ind] = 0
#    net_mat[output_ind] = 0
#    if p_flag:
#        net_mat[p_l_ind] = p_l_val
#        
#    # FIX: randomize execution order
#    r_ord = np.random.choice(np.arange(num_node), size=num_node, replace=False)
#    c_ord = np.random.choice(np.arange(num_node), size=num_node, replace=False)        
#        
#    # Ensure at least one specie is listed per row and column
#    for i in r_ord:
#        # Row
#        if np.sum(net_mat[i]) < 1:
#            # No bueno on boundary output
#            if i in output_ind:
#                pass
#            else:
#                # No direct reaction between boundary species
#                if i < num_bound:
#                    ind_choose = np.arange(num_bound, num_node)
#                else:
#                    ind_choose = np.arange(num_input, num_node)
#                # Ensure no auto-regulation
#                ind_choose = ind_choose[(ind_choose != i)]
#                # Ensure no changes in persistant layer
#                if p_flag:
#                    if i in np.array(p_l_ind).T[:,0]:
#                        ind_choose = ind_choose[(ind_choose != np.array(
#                        p_l_ind).T[np.where(np.array(
#                        p_l_ind).T[:,0] == i)[0]][:,1])]
#                # If no option is available, randomly change one
#                if len(ind_choose) == 0:
#                    # Make sure not to revert changes made from previous 
#                    # iterations (thus resulting empty row or column)
#                    sum_test = np.greater(np.sum(net_mat[ind_choose],axis=1),1)
#                    ind_choose = ind_choose[sum_test]
#                    rand_ind = np.random.choice(ind_choose)
#                    net_mat[rand_ind, i] = 0
#                    net_mat[i, rand_ind] = 1
#                else:
#                    # Randomly choose from rest of the indexes with random size
#                    net_mat[i, np.random.choice(ind_choose)] = 1
#    
#    for i in c_ord:
#        # Column
#        if np.sum(net_mat[:,i]) < 1:
#            # No bueno on boundary input
#            if i in input_ind:
#                pass
#            else:
#                # No direct reaction between boundary species
#                if i < num_bound:
#                    ind_choose = np.arange(num_bound, num_node)
#                else:
#                    ind_choose = np.append(input_ind, np.arange(num_bound, 
#                                                                num_node))
#                # Ensure no auto-regulation
#                ind_choose = ind_choose[(ind_choose != i)]
#                # Ensure no changes in persistant layer
#                if p_flag:
#                    if i in np.array(p_l_ind).T[:,1]:
#                        ind_choose = ind_choose[(ind_choose != np.array(
#                        p_l_ind).T[np.where(np.array(
#                        p_l_ind).T[:,1] == i)[0]][:,0])]
#                # If no option is available, randomly change one
#                if len(ind_choose) == 0:
#                    # Make sure not to revert changes made from previous 
#                    # iterations (thus resulting empty row or column)
#                    sum_test = np.greater(np.sum(net_mat[ind_choose],axis=1),1)
#                    ind_choose = ind_choose[sum_test]
#                    rand_ind = np.random.choice(ind_choose)
#                    net_mat[i, rand_ind] = 0
#                    net_mat[rand_ind, i] = 1
#                else:
#                    # Randomly choose from rest of the indexes with random size
#                    net_mat[np.random.choice(ind_choose), i] = 1   
#
#    if p_flag:
#        net_mat[p_l_ind] = p_l_val
#
#    return net_mat

def generateNetworkFromAntimony(ant_str):
    """
    Generate network matrix from Antimony string
    
    :param ant_str: Antimony string
    :rtype: array
    """
    
    import antimony
    import tellurium as te
    
    antimony.clearPreviousLoads()
    
    antimony.loadAntimonyString(ant_str)
    
    module = antimony.getModuleNames()[-1]
       
    num_rxn = int(antimony.getNumReactions(module))
    
    rct = antimony.getReactantNames(module)
    rct = [list(i) for i in rct]
    rct_flat = [item for sublist in rct for item in sublist]
    
    prd = antimony.getProductNames(module)
    prd = [list(i) for i in prd]
    prd_flat = [item for sublist in prd for item in sublist]
    
    ratelaw = antimony.getReactionRates(module)
    
    r = te.loada(ant_str)
    bnd = r.getBoundarySpeciesIds()
    flt = r.getFloatingSpeciesIds()
    s_arr = np.array(sorted(bnd + flt))
    
    ref_list = generateCombinationOfSpecies(s_arr)
    
    net_mat = np.zeros([len(ref_list), len(ref_list)])
    
    for i in range(num_rxn):
        if len(rct[i]) == 1 and len(prd[i]) == 1:
            r_s = rct[i][0]
            p_s = prd[i][0]
            r_s_i = ref_list.index(r_s)
            p_s_i = ref_list.index(p_s)
            net_mat[r_s_i][p_s_i] = 1
        elif len(rct[i]) == 2 and len(prd[i]) == 1:
            r_s1 = rct[i][0]
            r_s2 = rct[i][1]
            p_s = prd[i][0]
            r_s_i = ref_list.index(tuple(sorted((r_s1, r_s2))))
            p_s_i = ref_list.index(p_s)
            net_mat[r_s_i][p_s_i] = 1
        elif len(rct[i]) == 1 and len(prd[i]) == 2:
            r_s = rct[i][0]
            p_s1 = prd[i][0]
            p_s2 = prd[i][1]
            r_s_i = ref_list.index(r_s)
            p_s_i = ref_list.index(tuple(sorted((p_s1, p_s2))))
            net_mat[r_s_i][p_s_i] = 1
        elif len(rct[i]) == 2 and len(prd[i]) == 2:
            r_s1 = rct[i][0]
            r_s2 = rct[i][1]
            p_s1 = prd[i][0]
            p_s2 = prd[i][1]
            r_s_i = ref_list.index(tuple(sorted((r_s1, r_s2))))
            p_s_i = ref_list.index(tuple(sorted((p_s1, p_s2))))
            if '/' in ratelaw[i]: # Inhibition
                net_mat[r_s_i][p_s_i] = 2
            else:
                net_mat[r_s_i][p_s_i] = 1
    
    antimony.clearPreviousLoads()
    
    return net_mat, ref_list

def generateNetworkFromSBML(sbml):
    """
    Generate network matrix from a given model
    
    :param sbml: sbml string or file path
    :rtype: array
    """
    
    import roadrunner
    import tellurium as te
    
    try:
        ant_str = te.sbmlToAntimony(sbml)
    except:
        raise Exception("Cannot load SBML file/string. Check whether the \
                        filepath or SBML string is formatted correctly.")
    
    net_mat, ref_list = generateNetworkFromAntimony(ant_str)

    return net_mat, ref_list

def generateNetworkWithSteadyState(num_node, num_bound, num_input, knownr=1):
    """
    Generate network topology that has steady state solution
    
    :param num_node: total number of species
    :param num_bound: total number of boundary species
    :param num_input: number of input boundary species
    :rtype: array
    """
    import antimony
    import tellurium as te
    
    i = 0
    while i < 2: 
        try:
            test_net = generateRandomNetwork(num_node, num_bound, num_input)
            k_count = np.count_nonzero(test_net)
            s_init = generateInitialSpecies(num_node, 5)
            k_init = generateInitialRateConstants(k_count, 1)
            ant_str = generateAntimony(test_net, s_init, k_init, 
                                                num_node, num_bound)
    
            antimony.clearPreviousLoads()
            rr = te.loada(ant_str)
            rr.resetToOrigin() # ALWAYS RESET
            rr.steadyStateNamedArray() # Test steady state
            i += 1
        except ValueError:
            pass
        except RuntimeError:
            pass
        
    antimony.clearPreviousLoads()
        
    return test_net, s_init, k_init

def generateReducedPersistanceMatrix(p_net, num_node, c_arr):
    """
    Generate Persistance layer with restricted space.
    
    :param p_net: persistance matrix
    :param num_node: total number of species
    :param c_arr: combinations of species
    """
    for i in range(len(p_net)):
        for j in range(len(p_net[i])):
            # No auto-regulation
            if i == j:
                p_net[i][j] = 0
            if i >= num_node:
                if j >= num_node:
                    if (j == (len(p_net) - 1 - (i - num_node))):
                        p_net[i][j] = 0
            # No direct interaction between boundary species
            if 'I' in ''.join(c_arr[i]) and 'X' in ''.join(c_arr[j]):
                p_net[i][j] = 0
            # Make sure boundary species stay as input or output
            if 'I' in ''.join(c_arr[j]):
                p_net[:,j] = 0
            if 'X' in ''.join(c_arr[i]):
                p_net[i,:] = 0
                
    return p_net
    
def generateSpeciesList(num_float, num_input, num_output):
    """
    Generate a string list of species with boundary inputs and outputs marked.
    
    :param num_float: number of floating species
    :param num_input: number of input boundary species
    :param num_output: number of output boundary species
    """
    
    s_arr = np.empty(num_float + num_input + num_output).astype('str')
    
    for i in range(num_input):
        s_arr[i] =  ("$I" + str(i))
    for i in range(num_float): # Generate string array of all species
        s_arr[i + num_input] = ("S" + str(i))
    for i in range(num_output):
        s_arr[i + num_input + num_float] =  ("$X" + str(i))

    return s_arr

def generateSpeciesIndex(s_arr, c_arr):
    """
    Generate index to see which combination contains specific species.
    
    :param s_arr: total number of species
    :param c_arr: combinations of species
    """
    
    s_ind = []
    for i in range(len(s_arr)):
        s_ind_tmp = []
        for j in range(len(c_arr)):
            if s_arr[i] in c_arr[j]:
                s_ind_tmp.append(j)
        
        s_ind.append(s_ind_tmp)
    
    return s_ind
            
    