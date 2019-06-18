# -*- coding: utf-8 -*-

# COPYRIGHT 2017 Kiri Choi
# Script to generate, classify, and cluster networks by use of simple 
# perturbations

from __future__ import print_function

import tellurium as te
import numpy as np
import roadrunner
import antimony
import multiprocessing as mp
import psutil
from functools import partial
import settings
import clustering
import generate
import util
import truth
import itertools
import time
import sys

def simulationSandbox(setup, ant_str, r_comb):
    '''
    Function to parallelize simulations due to memory leak issue.
    Sandbox anything that has to do with antimony and kill it off after an
    iteration
    '''
    antimony.clearPreviousLoads()
    
    rr = te.loada(ant_str)
    rr.resetAll() # ALWAYS RESET
    rr.conservedMoietyAnalysis = True
    pert_i = rr.steadyStateNamedArray() # Initial steady state    
    
    # Put initial steady state and species ids
    #pert_init.append(pert_i[0])
        
    # Pertubation for rate constants
    k_pert_output_i = np.empty([len(r_comb), setup.num_float])
        
    for i in range(len(r_comb)):
        k_pert_output_i[i] = util.perturbRate(rr, r_comb[i], setup.k_pert)    

    antimony.clearPreviousLoads()

    return pert_i.tolist(), k_pert_output_i.tolist()

# Perform classification 
#@profile
def syn_clsfy(index):
    
    net_mat = generate.generateRandomNetworkNew(s_arr, c_arr, s_ind, p_net_c)

    if util.isUnique(net_mat, net_mat_list):
        num_r = util.getNumReactions(net_mat)
        r_ind = util.getPersistantOrder(net_mat, setup.p_net)
        s_init = generate.generateInitialSpecies(setup.num_node, setup.s_max, 
                                                 setup.p_s)
        k_init = generate.generateInitialRateConstants(num_r, setup.k_max, 
                                                       r_ind, setup.p_k)
        ant_str = generate.generateAntimonyNew(net_mat, s_init, k_init, 
                                            s_arr, c_arr)
    
        #net_mat_list[index] = net_mat
        #s_init_list.append(s_init)
        #k_init_list.append(k_init)
        
        # TODO: Combinations of perturbations
        r_comb = clustering.getListOfCombinations(r_ind)
        
        # TODO: Fix multiprocessing 
        # Fix for memory leak in Antimony
        
        #p = mp.Pool(processes=1, maxtasksperchild=100)
        #p_sandbox = partial(simulationSandbox, setup=setup, ant_str=ant_str, 
        #                    r_comb=r_comb)
        #pert_i, k_pert_output_i = p.map(p_sandbox, range(1))
        #p.close()
        #p.terminate()
        
        pert_i, k_pert_output_i = simulationSandbox(setup, ant_str, r_comb)
        
        # Workaround for issue with pickling NamedArray
        pert_i = np.array(pert_i)
        k_pert_output_i = np.array(k_pert_output_i)
        
        # Classification for rate constants
        k_class_i = np.empty([len(r_comb), setup.num_float], dtype=int)
        
        for i in range(len(r_comb)):
            for j in range(setup.num_float):
                k_diff = k_pert_output_i[i][j] - pert_i[0][j]
                if (np.abs(k_diff) > setup.Thres):
                    if k_diff < 0.:
                        k_class_i[i][j] = 1
                    else:
                        k_class_i[i][j] = 2
                else:
                    k_class_i[i][j] = 0
        
        # TODO: Only collect those that matches the output of the true network
        # This might be more memory intensive
        if truth.compareIndClass(t_analysis[2], k_class_i):
            net_mat_list.append(net_mat)
            #k_pert_output.append(k_pert_output_i)
            k_class.append(k_class_i)
        
    #Progress
    if int((index + 1)/100.) == (index + 1)/100.:
        print('run ' + str(index + 1) + ', memory: ' + 
                                                  str(psutil.virtual_memory()))
        
    if int((index + 1)/1000.) == (index + 1)/1000.:
        print('collected ' + str(len(net_mat_list)) + 
              ' functional networks so far; Delta_t: ' + str(time.time() - t1))
    
#---- Begin executable --------------------------------------------------------
if __name__ == '__main__':
    
    t1 = time.time() # Performance testing
    roadrunner.Config.setValue(roadrunner.Config.ROADRUNNER_DISABLE_WARNINGS,3)
    
    # Set seed
    r_seed = None
    
    np.random.seed(r_seed)
    
    # Initialize
    setup = settings.import_settings()
    s_arr = generate.generateSpeciesList(setup.num_float, setup.num_input, setup.num_output)
    c_arr = generate.generateCombinationOfSpecies(s_arr)
    s_ind = generate.generateSpeciesIndex(s_arr, c_arr)
    p_net_c = generate.generateReducedPersistanceMatrix(setup.p_net, setup.num_node, c_arr)
    
    # Ground truth analysis
    t_analysis = truth.classify(setup, s_arr, c_arr)
    
    # No need to resort--------------------------------------------------------
    # Sorting based on species index
    #t_analysis[1][0] = util.resort(t_analysis[1][0], t_analysis[3][0])
    #t_analysis[2][0] = util.resort(t_analysis[2][0], t_analysis[3][0])
    # -------------------------------------------------------------------------

    # List of network matrices
    #net_mat_list = np.empty([num_sets, num_node, num_node]) 
    net_mat_list = []
    
    #s_init_list = [] # List of initial concentrations
    #k_init_list = [] # List of initial rate constants
    #species_list = [] # List of floating species
    
    # Lists for steady state output values and its boolean representation 
    # against initial steady states for both perturbations on reactions and
    # species.
    # TODO: This is obviously memory intensive.
    #pert_init = [] # List of initial steady state values without perturbation
    #k_pert_output = []
    k_class = []
    
    # Not used anymore --------------------------------------------------------
    #p_r_ind = util.getPersistantIndex(setup.p_layer)
    # -------------------------------------------------------------------------
    
    # TODO: Issues with certain network topology with steady state calculation.
    # Ignore the issue and rely on RNG for now
    # TODO: Way to seletively choose speicies for steady state solutions?
    print("Starting iteration...")
    
    i = 0
    while i < setup.num_sets:
        try:
            syn_clsfy(i)
            i += 1
        except ValueError:
            pass
        except RuntimeError:
            pass
    
    if len(net_mat_list) == 0:
        raise Exception('No possible solution found. Perhaps increase the '
                        'number of iterations?')

    net_mat_list = np.array(net_mat_list)
    
    # No need to resort--------------------------------------------------------
    # Sorting based on species index
    #for i in range(num_sets):
    #    k_pert_output[i] = util.resort(k_pert_output[i], species_list[i])
    #    k_class[i] = util.resort(k_class[i], species_list[i])
    
    # Return unique network matrices and its index
    #u_net_mat, u_ind = util.getUniqueNetwork(net_mat_list, num_sets, num_node)
    # -------------------------------------------------------------------------

# Now the code only calculates reaction defined in p_layer---------------------
#    
#    # Generate num_node x num_node matrix where each element at index (i,j) 
#    # contains a list of indices of network matrices that have a reaction a 
#    # said index
#     net_ind_group = util.groupReaction(net_mat_list)
#
#    # Sort boolean perturbation outputs in num_node x num_node matrix based on 
#    # net_ind_group. Looks for correct index by searching the reaction orders
#    sorted_k_class = []
#    for i in range(len(net_ind_group)):
#        s_k_c_2 = []
#        for j in range(len(net_ind_group[i])):
#            s_k_c_1 = []
#            if len(net_ind_group[i][j]) != 0:
#                for k in range(len(net_ind_group[i][j])):
#                    r_ind = np.where((np.array(np.where(
#                            net_mat_list[net_ind_group[i][j][k]] == 1)
#                            ).T == (i, j)).all(axis=1))[0][0]
#                    s_k_c_1.append(k_class[net_ind_group[i][j][k]][r_ind])
#            s_k_c_2.append(s_k_c_1)
#        sorted_k_class.append(s_k_c_2)
# -----------------------------------------------------------------------------
    
# Maximum discrimination disabled ---------------------------------------------
#    # Generate combinations of perturbations
#    com_ind = []
#    for i in range(num_float):
#        com_ind.append(list(itertools.combinations(range(num_float), i + 1)))
#    
#    # Flatten
#    com_ind = [item for sublist in com_ind for item in sublist]
#    
#    # Generate clusters using all combinations of purturbations
#    cluster_output = []
#    for i in range(len(com_ind)):
#        cluster_output.append(clustering.run(sorted_k_class, 
#                                                     com_ind[i]))
#    
#    # Search for perturbations with maximum discrimitory potential
#    disc_list = clustering.maximumDisc(cluster_output)
#    
#    # Return indexes in network matrix with maximum discriminability with index
#    # for list of combination of perturbation
#    tar_r_ind = clustering.pinpoint(disc_list, 3)
# -----------------------------------------------------------------------------
    
#    t_net_ind, nt_net_ind = truth.compareClass(t_analysis, k_class)
    
# Deprecated since the output of true network is simplified -------------------
    #t_net_ind_c = clustering.serachCommon(t_net_ind)
# -----------------------------------------------------------------------------
    
#    t_net_mean = clustering.maximumFreq(net_mat_list, t_net_ind, nt_net_ind)

# Need to fix issue in line #74 -----------------------------------------------    
    t_net_mean = clustering.maximumFreqInd(net_mat_list)
# -----------------------------------------------------------------------------

    pd = util.getDataFrame(t_net_mean)
    
    print(time.time() - t1)
       
#    k_dist = np.empty([num_sets])
#    k_dist_per = []
#    s_dist = np.empty([num_sets])
#    
#    # Detect features
#    for i in range(num_sets):
#        k_dist[i] = distance(k_pert_output[i])
#        k_dist_per.append(distance_per_k(k_pert_output[i]))
#        s_dist[i] = distance(s_pert_output[i])
        
    #k_corr = corr_count(k_class)
    #s_corr = corr_count(s_class)
    

