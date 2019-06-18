# -*- coding: utf-8 -*-

# COPYRIGHT 2017 Kiri Choi
# Script to setup the clustering

from __future__ import print_function

import numpy as np

class import_settings():
    def __init__(self):
        
        self.num_float = 3 # Number of floating species
        self.num_input = 1 # Number of input bounary species
        self.num_output = 1 # Number of output boundary species
        self.num_bound = self.num_input + self.num_output 
        self.num_node = self.num_float + self.num_bound # Number of total nodes
        self.s_max = 1.0 # Maximum initial concentration of flaoting species
        self.k_max = 1.0 # Maximum value for rate constants
        self.num_sets = 10000 # Number of data sets
        
        # Perturbation
        self.s_pert = 1.0 # Perturbation amount for species concentrations
        self.k_pert = 1.0 # Perturbation amount for rate constants    
    
        # Classification
        self.Thres = 1e-3 # Threshold for boolean classification
    
        # Persistant layer
        # Persistant layer provides network generation algorithm specific reactions
        # to conserve while generating networks. None denotes no information.
        # Persistant network matrix
        
        
#        # Signaling cascade
#        self.p_net =   np.array([[ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                 [ 1,     None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                 [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                 [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                 [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                 [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                 [ None,  None,  None,  None,  None,  None,  None,  None,  2,     None],
#                                 [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                 [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                 [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None]])
#        
#        self.p_s = np.array([4., 2., 1., 0.8])
#        
#        self.p_k =       np.array([[ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                   [ 0.83,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                   [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                   [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                   [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                   [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                   [ None,  None,  None,  None,  None,  None,  None,  None,  0.5,   None],
#                                   [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                   [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                   [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None]])
#                        
#        self.t_net = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
#    
#        self.t_s = np.array([4., 2., 1., 0.8])
#        
#        self.t_k =   np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0.83, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0.95, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    
    
            
        
#        self.p_net = np.array([[None, 1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, 1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, 1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]])
#    
#        # Persistant species initial concentration
#        self.p_s = np.array([2.5, None, None, None, None, 2.0])
#        
#        # Persistant rate constant
#        self.p_k =   np.array([[None, .85, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, .35, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, .6, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]])
#        
#        # Analyze the truth
#        # Truth network matrix
#        self.t_net = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
#                          
#        # Truth initial species
#        self.t_s = np.array([2.5, .9, 1.25, 1.75, .7, 2.0])
#        
#        # Truth rate constants
#        self.t_k = np.array([[0, .85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                             [0, 0, .9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                             [0, .4, 0, 0, 0, .35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                             [0, 0, 0, .35, 0, .6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .85, 0, 0, 0, 0],
#                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        
    
    
    
        # C1-FFL
        self.p_net = np.array([[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
                               [None, None, None,    1, None, None, None, None, None, None, None, None, None, None, None],
                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
                               [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
                               [None, None, None, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [None, None, None, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [None, None, None, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [None, None, None, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [None, None, None, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [None, None, None, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [None, None, None, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [None, None, None, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [None, None, None, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [None, None, None, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    
        # Persistant species initial concentration
        self.p_s = np.array([2.5, .9, 1.25, 1.75, 2.])
        
        # Persistant rate constant
        self.p_k = np.array([[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
                             [None, None, None, 1.1 , None, None, None, None, None, None, None, None, None, None, None],
                             [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
                             [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
                             [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
                             [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
                             [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
                             [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
                             [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
                             [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
                             [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
                             [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
                             [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
                             [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
                             [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]])
        
        # Analyze the truth
        # Truth network matrix
        self.t_net = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                          
        # Truth initial species
        self.t_s = np.array([2.5, .9, 1.25, 1.75, 2.])
        
        # Truth rate constants
        self.t_k = np.array([[0, 0.85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0.8, 1.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    
    
#        #
#        self.p_layer = np.array([[0, 0, 1, 0, 0, 0],
#                                 [0, 0, 0, 0, 0, 0],
#                                 [0, 0, None, 1, None, None],
#                                 [0, 0, None, None, None, None],
#                                 [0, 0, None, None, None, 1],
#                                 [0, 1, None, None, 1, None]])
#    
#        self.p_s_layer = np.array([1., .1, .1, 0.7, 0.5, 0.2])
#        
#        self.p_k_layer = np.array([[None, None, 0.6, None, None, None],
#                                   [None, None, None, None, None, None],
#                                   [None, None, None, 0.7, None, None],
#                                   [None, None, None, None, None, None],
#                                   [None, None, None, None, None, 0.25],
#                                   [None, 0.75, None, None, 0.4, None]])    
#    
#        self.t_net = np.array([[ 0.,  0.,  1.,  0.,  0.,  0.],
#                                [ 0.,  0.,  0.,  0.,  0.,  0.],
#                                [ 0.,  0.,  0.,  1.,  0.,  0.],
#                                [ 0.,  0.,  1.,  0.,  1.,  0.],
#                                [ 0.,  0.,  0.,  0.,  0.,  1.],
#                                [ 0.,  1.,  0.,  0.,  1.,  0.]])
#
#        self.t_s = np.array([1., .1, 0.1, 0.7, 0.5, 0.2])
#        
#        self.t_k = np.array([[None, None, 0.6, None, None, None],
#                             [None, None, None, None, None, None],
#                             [None, None, None, 0.7, None, None],
#                             [None, None, 0.2, None, 0.2, None],
#                             [None, None, None, None, None, 0.25],
#                             [None, 0.75, None, None, 0.4, None]])

    
    
    
#        self.p_layer = np.array([[ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                [ None,  None,  None,  None,  None,  None,  None,  1,  None,  None],
#                                [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                [ None,  1,  None,  None,  None,  None,  None,  None,  None,  None],
#                                [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                [ None,  None,  None,  None,  None,  None,  None,  None,  1,  None]])
#        
#        self.p_s_layer = np.array([ 2.911489,  1.868356, None,  None,  None,  None,  None, None, None, None])
#        
#        self.p_k_layer = np.array([[ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                  [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                  [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                  [ None,  None,  None,  None,  None,  None,  None,  0.523175,  None,  None],
#                                  [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                  [ None,  0.209591,  None,  None,  None,  None,  None,  None,  None,  None],
#                                  [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                  [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                  [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                                  [ None,  None,  None,  None,  None,  None,  None,  None,  0.834612,  None]])
#                        
#        self.t_net = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#                              [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
#                              [0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
#                              [0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
#                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#                              [0, 0, 1, 1, 0, 0, 0, 0, 1, 0]])
#    
#        self.t_s = np.array([ 2.911489,  1.868356,  2.509696,  1.801193,  1.874685,  2.293218,
#            0.152692,  3.686649,  1.032535,  1.966019])
#        
#        self.t_k = np.array([[ None,  None,  None,  None,  None,  None,  None,  None,  0.239233,  None],
#                            [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None],
#                            [ None,  0.237564,  None,  None,  None,  None,  None,  None,  None,  None],
#                            [ None,  None,  None,  None,  None,  None,  None,   0.523175,  None,  None],
#                            [ None,  None,  None,  None,  None,  0.959494,  None,  0.207877,  None,  None],
#                            [ None,  0.209591,  None,  None,  None,  None,  None,  None,   0.161315, 0.710621 ],
#                            [ None,  None,  None,  None,  0.80108 ,  None,  None,  0.190327,  None,  0.416941 ],
#                            [ None,  None,  None,  None,  None,  None,  0.759073,  None,  None,  None],
#                            [ None,  None,  None,  None,  None,  None,  None,  None,  None,   0.141025],
#                            [ None,  None,  0.444994,  0.945646  ,  None,  None,  None,  None,  0.834612,  None]])  
    
        
    
    