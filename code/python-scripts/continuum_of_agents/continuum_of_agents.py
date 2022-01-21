#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The deep equilibrium net "continuum of agents" model: 

This script provides the code used to model and solve a model with a continuum of agents, and aggregate and
idiosyncratic shocks in appendix E of Azinovic, Gaegauf, & Scheidegger (2021)
(https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3393482). For a more streamlined application, see 
https://github.com/sischei/DeepEquilibriumNets/blob/master/code/jupyter-notebooks/analytic/Analytic_tf1.ipynb.

Note that, this script was programmed in TensorFlow 2 and is not TensorFlow 1 compatible. To install
Tensorflow 2, use 
> pip install tensorflow

To upgrade from TensorFlow 1 to TensorFlow 2, use
> pip install --upgrade tensorflow

-----------------------------------------------------------------------------------------------------
There are two modes to run this code: 1) the final network weights can be loaded and used to output a
host of plots; 2) the deep equilibrium net can be trained from scratch. We have simplified the code
such that the only user input is the desired running mode. To run, follow these instructions:

In terminal:
> cd '/DeepEquilibriumNets/code/python-scripts/continuum_of_agents'

Mode 1: Load the trained network weights
> python continuum_of_agents.py
The results are saved to ./output/deqn_continuumagents_restart

Mode 2: Train from scratch
> python continuum_of_agents.py --train_from_scratch
The results are saved to ./output/deqn_continuumagents
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

print('tf version:', tf.__version__)

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')

plt.rcParams.update({'font.size': 12})

perc_list = [10, 50, 90, 99.9]
perc_ls = {0.1 : '--', 10 : '-.', 50 : ':', 90 : '-.', 99.9 : '--'}
std_figsize = (4, 4)

print_more_error_stats = False # turn on to plot more error statistics (high memory usage)
print_more_stats_interval = 1000 # interval for which to print more error statistics


def train(
    seed, lr, optimizer_name, 
    num_hidden_nodes_price, activations_hidden_nodes_price, 
    num_hidden_nodes_pol, activations_hidden_nodes_pol, 
    batch_size, num_episodes, len_episodes, epochs_per_episode, num_tracks, num_id_per_shock,
    path_wd, run_name,
    save_interval,
    train_flag, load_bool, load_run_name, load_episode):

    #=========================================================================================================================================
    # Initialize parameters
    #=========================================================================================================================================
    # Set the seed
    np.random.seed(seed)
    tf.random.set_seed(seed)

    nZ = 6 # number of aggregate shocks
    assert nZ == 6, 'hard coded'
    nEta = 2 # number of idiosyncractic shocks
    assert nEta == 2, 'hard coded'

    nUnc = 2
    nIn = 3

    assert nZ == nUnc * nIn, 'number states doesnt fit'

    pers_unc = np.array([
        [0.8, 0.2],
        [0.2, 0.8]
    ])

    pers_in_normal = np.array(
    [
        [0.70, 0.20, 0.10],
        [0.15, 0.70, 0.15],
        [0.10, 0.20, 0.70]
    ])

    pers_in_high = np.array(
    [
        [0.75, 0.10, 0.15],
        [0.20, 0.60, 0.20],
        [0.15, 0.10, 0.75]
    ])

    pi_Z_np = np.empty((nZ, nZ))

    assert nUnc == 2, 'hardcoded in transition matrix below'

    pi_Z_np[0 : nIn, 0 : nIn] = pers_in_normal * pers_unc[0, 0]
    pi_Z_np[0 : nIn, nIn : 2 * nIn] = pers_in_normal * pers_unc[0, 1]
    pi_Z_np[nIn : 2 * nIn, 0 : nIn] = pers_in_high * pers_unc[1, 0]
    pi_Z_np[nIn : 2 * nIn, nIn : 2 * nIn] = pers_in_high * pers_unc[1, 1]

    pi_Z = tf.constant(pi_Z_np , dtype=tf.float32) # Transition probabilities for the aggregate shocks
    Y_vec = tf.constant([[0.95], [1.00], [1.05], [0.95], [1.00], [1.05]], dtype=tf.float32) # Output depending on the aggregate shock
    wage_vec = Y_vec  # Wage depending on the aggregate shock

    pi_Eta = tf.constant(np.array([[0.7, 0.3], [0.3, 0.7]]), dtype=tf.float32)  # Transition probabilities idiosyncratic shocks
    eta_vec = tf.constant([[0.8], [1.2]], dtype=tf.float32)                     # Idiosyncratic labor endowment depending on id. shock

    nA = 100                             # Number of histogram points
    amin, amax = 0.0, 20.0               # Upper and lower bound on asset holding
    agrid = tf.linspace(amin, amax, nA)  # Asset grid for the histogram
    deltaa = agrid[1] - agrid[0]         # Grid step
    sigma = 8.0                          # Risk aversion
    rho = 2.0                            # 1 / IES
    beta = 0.95                          # Discount factor

    # build vector of idiosyncratic  states
    id_vec_np = np.zeros([2 * nA, 2])
    id_vec_np[:nA, 0] = 0.
    id_vec_np[:nA, 1] = agrid.numpy()
    id_vec_np[nA :, 0] = 1.0
    id_vec_np[nA :, 1] = agrid.numpy()
    id_vec = tf.constant(id_vec_np, dtype=tf.float32)

    print('pi_Z_np = ', pi_Z_np)
    print('Y_vec = ', Y_vec)
    print('wage_vec = ', wage_vec)
    print('pi_Eta = ', pi_Eta)
    print('eta_vec = ', eta_vec)
    print('nA = ', nA)
    print('agrid = ', agrid)
    print('deltaa = ', deltaa)
    print('sigma = ', sigma)
    print('rho = ', rho)
    print('beta = ', beta)

    output_path = os.path.join(path_wd, 'output')
    save_base_path = os.path.join(output_path, run_name)
    plot_path = os.path.join(save_base_path, 'plots')
    save_path = os.path.join(save_base_path, 'model')
    
    if 'output' not in os.listdir(path_wd):
        os.mkdir(output_path)

    if run_name not in os.listdir(output_path):
        os.mkdir(save_base_path)
        os.mkdir(plot_path)
        os.mkdir(save_path)

    save_name_agrid = os.path.join(save_path, 'agrid.npy')
    np.save(save_name_agrid, agrid)
    print('agrid saved to', save_name_agrid)          

    #=========================================================================================================================================
    # Transition of the aggregate state
    #=========================================================================================================================================
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32), 
        tf.TensorSpec(shape=(nA,), dtype=tf.float32), 
        tf.TensorSpec(shape=(), dtype=tf.float32)])
    def get_weights_grid(asearch, agrid, deltaa):
        assert asearch.shape[1] == 1, 'asearch should be of shape (?, 1)'

        agrid_2 = tf.reshape(agrid, [1, -1])
        dif = tf.math.abs(agrid_2 - asearch)
        mask = tf.cast((dif < deltaa), dtype=tf.float32)
        w1 = 1.0 - (dif / deltaa) 
        w1 = w1 * mask
        w1 = w1 / tf.math.reduce_sum(w1, axis=1, keepdims=True)
  
        return w1


    @tf.function
    def get_weights_eta(net, X_agg):
        nAgg = X_agg.shape[0] # number of aggregate state

        # construct the state for all eta_t and all a_t on the histogram grid
        X_agg_rep = tf.repeat(X_agg, nEta * nA * tf.ones(nAgg, dtype=tf.int32), axis=0)
        X_id_rep = tf.tile(id_vec, [nAgg, 1])
        X = tf.concat([X_id_rep, X_agg_rep], axis=1) # of shape (nAgg * (nEta * nA), 2 * nA + 3) 
        
        # evaluate the neural net to get the policy
        anext = net(X)[:, 0:1] # of shape (nAgg * (nEta * nA), 1)
        anext = tf.minimum(anext, amax * tf.ones_like(anext))

        # get weights associated with anext
        assoc_weightsnext = get_weights_grid(anext, agrid, deltaa) # of shape (nAgg * (nEta * nA), nA)
        assoc_weightsnext = tf.reshape(assoc_weightsnext, [nAgg, nEta * nA, nA])
        
        # weights for each etanext
        new_weights_list = []
        
        for etaidx in range(nEta):
            from_list = []
            weight_old = tf.expand_dims(X_agg[:, 3 + etaidx * nA : 3 + (etaidx + 1) * nA], -1) # get h
            temp_weights_new = assoc_weightsnext[:, etaidx * nA : (etaidx + 1) * nA, :]
            for etanidx in range(nEta):
                trans_prob = pi_Eta[etaidx, etanidx]
                weight_etatransition = trans_prob * tf.math.reduce_sum(weight_old * temp_weights_new, axis=1)
                from_list.append(weight_etatransition)
            new_weights_list.append(from_list)
        
        new_weights = tf.concat([sum(x) for x in zip(*new_weights_list)], axis=1)
        
        return new_weights

    @tf.function
    def get_Xagg_next(net, X_agg):
        nAgg = X_agg.shape[0]
        
        zidxnext0 = 0.0 * tf.ones([nAgg, 1])
        zidxnext1 = 1.0 * tf.ones([nAgg, 1])
        zidxnext2 = 2.0 * tf.ones([nAgg, 1])
        zidxnext3 = 3.0 * tf.ones([nAgg, 1])
        zidxnext4 = 4.0 * tf.ones([nAgg, 1])
        zidxnext5 = 5.0 * tf.ones([nAgg, 1])
        
        inidxnext0 = 0.0 * tf.ones([nAgg, 1])
        inidxnext1 = 1.0 * tf.ones([nAgg, 1])
        inidxnext2 = 2.0 * tf.ones([nAgg, 1])
        inidxnext3 = 0.0 * tf.ones([nAgg, 1])
        inidxnext4 = 1.0 * tf.ones([nAgg, 1])
        inidxnext5 = 2.0 * tf.ones([nAgg, 1])
        
        uncidxnext0 = 0.0 * tf.ones([nAgg, 1])
        uncidxnext1 = 0.0 * tf.ones([nAgg, 1])
        uncidxnext2 = 0.0 * tf.ones([nAgg, 1])
        uncidxnext3 = 1.0 * tf.ones([nAgg, 1])
        uncidxnext4 = 1.0 * tf.ones([nAgg, 1])
        uncidxnext5 = 1.0 * tf.ones([nAgg, 1])    
        
        weightsnext = get_weights_eta(net, X_agg)
        
        Xaggnext0 = tf.concat([zidxnext0, inidxnext0, uncidxnext0, weightsnext], axis=1)
        Xaggnext1 = tf.concat([zidxnext1, inidxnext1, uncidxnext1, weightsnext], axis=1)
        Xaggnext2 = tf.concat([zidxnext2, inidxnext2, uncidxnext2, weightsnext], axis=1)
        Xaggnext3 = tf.concat([zidxnext3, inidxnext3, uncidxnext3, weightsnext], axis=1)   
        Xaggnext4 = tf.concat([zidxnext4, inidxnext4, uncidxnext4, weightsnext], axis=1)
        Xaggnext5 = tf.concat([zidxnext5, inidxnext5, uncidxnext5, weightsnext], axis=1)

        return Xaggnext0, Xaggnext1, Xaggnext2, Xaggnext3, Xaggnext4, Xaggnext5

    #=========================================================================================================================================
    # Cost function
    #=========================================================================================================================================
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 3 + 2 * nA), dtype=tf.float32)])
    def get_zidxwage(X_agg):
        zidx = tf.cast(X_agg[:, 0:1], tf.int32)
        wage = tf.gather(wage_vec, zidx[:, 0])
        
        return zidx, wage

    @tf.function
    def cost(net_pol, net_price, X, Xaggnext0, Xaggnext1, Xaggnext2, Xaggnext3, Xaggnext4, Xaggnext5):
        eps = 1e-6
        
        nSamples = X.shape[0]
        
        ### read out the state
        # idiosyncratic state
        etaidx = tf.cast(X[:, 0:1], tf.int32)
        eta = tf.gather(eta_vec, etaidx[:, 0])
        a = X[:, 1:2]
        
        # aggregate state
        X_agg = X[:, 2 :]
        
        ### compute aggregate quantities today
        zidx, wage = get_zidxwage(X_agg)
        
        p = net_price(X_agg)
        
        ### compute consumption today
        # get forecasts policy
        prediction = net_pol(X)
        anext = prediction[:, 0:1]
        lambd = prediction[:, 1:2]
        V = prediction[:, 2:3]
        # get consumption from budget constraint
        c_try = a + eta * wage - anext * p
        c = tf.maximum(c_try, eps * tf.ones_like(c_try))
        
        ### construct next periods states for all combinations of aggregate and idiosyncratic shocks
        # aggregate state next period
        
        _, wagenext0 = get_zidxwage(Xaggnext0)
        _, wagenext1 = get_zidxwage(Xaggnext1)
        _, wagenext2 = get_zidxwage(Xaggnext2)
        _, wagenext3 = get_zidxwage(Xaggnext3)
        _, wagenext4 = get_zidxwage(Xaggnext4)
        _, wagenext5 = get_zidxwage(Xaggnext5)
        
        pnext0 = net_price(Xaggnext0)
        pnext1 = net_price(Xaggnext1)
        pnext2 = net_price(Xaggnext2)
        pnext3 = net_price(Xaggnext3)
        pnext4 = net_price(Xaggnext4)
        pnext5 = net_price(Xaggnext5)    

        # idiosyncratic state
        etaidxnext0 = 0. * tf.ones([nSamples, 1], dtype=tf.float32)
        etanext0 = eta_vec[0, 0]
        etaidxnext1 = 1. * tf.ones([nSamples, 1], dtype=tf.float32)
        etanext1 = eta_vec[1, 0]
        
        Xidnext0 = tf.concat([etaidxnext0, anext], axis=1)
        Xidnext1 = tf.concat([etaidxnext1, anext], axis=1)
        
        # states next period
        Xnext_agg0_id0 = tf.concat([Xidnext0, Xaggnext0], axis=1)
        Xnext_agg0_id1 = tf.concat([Xidnext1, Xaggnext0], axis=1)
        Xnext_agg1_id0 = tf.concat([Xidnext0, Xaggnext1], axis=1)
        Xnext_agg1_id1 = tf.concat([Xidnext1, Xaggnext1], axis=1)
        Xnext_agg2_id0 = tf.concat([Xidnext0, Xaggnext2], axis=1)
        Xnext_agg2_id1 = tf.concat([Xidnext1, Xaggnext2], axis=1)
        Xnext_agg3_id0 = tf.concat([Xidnext0, Xaggnext3], axis=1)
        Xnext_agg3_id1 = tf.concat([Xidnext1, Xaggnext3], axis=1)
        Xnext_agg4_id0 = tf.concat([Xidnext0, Xaggnext4], axis=1)
        Xnext_agg4_id1 = tf.concat([Xidnext1, Xaggnext4], axis=1)
        Xnext_agg5_id0 = tf.concat([Xidnext0, Xaggnext5], axis=1)
        Xnext_agg5_id1 = tf.concat([Xidnext1, Xaggnext5], axis=1)    
        
        ### get next periods consumption
        
        # get forecasts policy
        prediction_a0_i0 = net_pol(Xnext_agg0_id0)
        prediction_a0_i1 = net_pol(Xnext_agg0_id1)
        prediction_a1_i0 = net_pol(Xnext_agg1_id0)
        prediction_a1_i1 = net_pol(Xnext_agg1_id1)
        prediction_a2_i0 = net_pol(Xnext_agg2_id0)
        prediction_a2_i1 = net_pol(Xnext_agg2_id1)
        prediction_a3_i0 = net_pol(Xnext_agg3_id0)
        prediction_a3_i1 = net_pol(Xnext_agg3_id1)
        prediction_a4_i0 = net_pol(Xnext_agg4_id0)
        prediction_a4_i1 = net_pol(Xnext_agg4_id1)
        prediction_a5_i0 = net_pol(Xnext_agg5_id0)
        prediction_a5_i1 = net_pol(Xnext_agg5_id1)

        anextnext_a0_i0 = prediction_a0_i0[:, 0:1]
        anextnext_a0_i1 = prediction_a0_i1[:, 0:1]
        anextnext_a1_i0 = prediction_a1_i0[:, 0:1]
        anextnext_a1_i1 = prediction_a1_i1[:, 0:1]
        anextnext_a2_i0 = prediction_a2_i0[:, 0:1]
        anextnext_a2_i1 = prediction_a2_i1[:, 0:1]
        anextnext_a3_i0 = prediction_a3_i0[:, 0:1]
        anextnext_a3_i1 = prediction_a3_i1[:, 0:1]
        anextnext_a4_i0 = prediction_a4_i0[:, 0:1]
        anextnext_a4_i1 = prediction_a4_i1[:, 0:1]
        anextnext_a5_i0 = prediction_a5_i0[:, 0:1]
        anextnext_a5_i1 = prediction_a5_i1[:, 0:1]    
        
        Vnext_a0_i0 = prediction_a0_i0[:, 2:3]
        Vnext_a0_i1 = prediction_a0_i1[:, 2:3]
        Vnext_a1_i0 = prediction_a1_i0[:, 2:3]
        Vnext_a1_i1 = prediction_a1_i1[:, 2:3]
        Vnext_a2_i0 = prediction_a2_i0[:, 2:3]
        Vnext_a2_i1 = prediction_a2_i1[:, 2:3]
        Vnext_a3_i0 = prediction_a3_i0[:, 2:3]
        Vnext_a3_i1 = prediction_a3_i1[:, 2:3]
        Vnext_a4_i0 = prediction_a4_i0[:, 2:3]
        Vnext_a4_i1 = prediction_a4_i1[:, 2:3]
        Vnext_a5_i0 = prediction_a5_i0[:, 2:3]
        Vnext_a5_i1 = prediction_a5_i1[:, 2:3]
        
        cnext_a0_i0_try = anext + etanext0 * wagenext0 - anextnext_a0_i0 * (pnext0)
        cnext_a0_i1_try = anext + etanext1 * wagenext0 - anextnext_a0_i1 * (pnext0)
        cnext_a1_i0_try = anext + etanext0 * wagenext1 - anextnext_a1_i0 * (pnext1)
        cnext_a1_i1_try = anext + etanext1 * wagenext1 - anextnext_a1_i1 * (pnext1)
        cnext_a2_i0_try = anext + etanext0 * wagenext2 - anextnext_a2_i0 * (pnext2)
        cnext_a2_i1_try = anext + etanext1 * wagenext2 - anextnext_a2_i1 * (pnext2)
        cnext_a3_i0_try = anext + etanext0 * wagenext3 - anextnext_a3_i0 * (pnext3)
        cnext_a3_i1_try = anext + etanext1 * wagenext3 - anextnext_a3_i1 * (pnext3)
        cnext_a4_i0_try = anext + etanext0 * wagenext4 - anextnext_a4_i0 * (pnext4)
        cnext_a4_i1_try = anext + etanext1 * wagenext4 - anextnext_a4_i1 * (pnext4)
        cnext_a5_i0_try = anext + etanext0 * wagenext5 - anextnext_a5_i0 * (pnext5)
        cnext_a5_i1_try = anext + etanext1 * wagenext5 - anextnext_a5_i1 * (pnext5)
        
        cnext_a0_i0 = tf.maximum(cnext_a0_i0_try, eps * tf.ones_like(cnext_a0_i0_try, dtype=tf.float32))
        cnext_a0_i1 = tf.maximum(cnext_a0_i1_try, eps * tf.ones_like(cnext_a0_i1_try, dtype=tf.float32))
        cnext_a1_i0 = tf.maximum(cnext_a1_i0_try, eps * tf.ones_like(cnext_a1_i0_try, dtype=tf.float32))
        cnext_a1_i1 = tf.maximum(cnext_a1_i1_try, eps * tf.ones_like(cnext_a1_i1_try, dtype=tf.float32))
        cnext_a2_i0 = tf.maximum(cnext_a2_i0_try, eps * tf.ones_like(cnext_a2_i0_try, dtype=tf.float32))
        cnext_a2_i1 = tf.maximum(cnext_a2_i1_try, eps * tf.ones_like(cnext_a2_i1_try, dtype=tf.float32))
        cnext_a3_i0 = tf.maximum(cnext_a3_i0_try, eps * tf.ones_like(cnext_a3_i0_try, dtype=tf.float32))
        cnext_a3_i1 = tf.maximum(cnext_a3_i1_try, eps * tf.ones_like(cnext_a3_i1_try, dtype=tf.float32))    
        cnext_a4_i0 = tf.maximum(cnext_a4_i0_try, eps * tf.ones_like(cnext_a4_i0_try, dtype=tf.float32))
        cnext_a4_i1 = tf.maximum(cnext_a4_i1_try, eps * tf.ones_like(cnext_a4_i1_try, dtype=tf.float32))
        cnext_a5_i0 = tf.maximum(cnext_a5_i0_try, eps * tf.ones_like(cnext_a5_i0_try, dtype=tf.float32))
        cnext_a5_i1 = tf.maximum(cnext_a5_i1_try, eps * tf.ones_like(cnext_a5_i1_try, dtype=tf.float32))    
        
        ### get transition probabilities
        idnext0 = tf.gather(pi_Eta, etaidx[:, 0])[:, 0:1]
        idnext1 = tf.gather(pi_Eta, etaidx[:, 0])[:, 1:2]
        
        aggnext0 = tf.gather(pi_Z, zidx[:, 0])[:, 0:1]
        aggnext1 = tf.gather(pi_Z, zidx[:, 0])[:, 1:2]
        aggnext2 = tf.gather(pi_Z, zidx[:, 0])[:, 2:3]
        aggnext3 = tf.gather(pi_Z, zidx[:, 0])[:, 3:4]
        aggnext4 = tf.gather(pi_Z, zidx[:, 0])[:, 4:5]
        aggnext5 = tf.gather(pi_Z, zidx[:, 0])[:, 5:6]
        
        trans_a0i0 = aggnext0 * idnext0
        trans_a0i1 = aggnext0 * idnext1
        trans_a1i0 = aggnext1 * idnext0
        trans_a1i1 = aggnext1 * idnext1
        trans_a2i0 = aggnext2 * idnext0
        trans_a2i1 = aggnext2 * idnext1
        trans_a3i0 = aggnext3 * idnext0
        trans_a3i1 = aggnext3 * idnext1    
        trans_a4i0 = aggnext4 * idnext0
        trans_a4i1 = aggnext4 * idnext1
        trans_a5i0 = aggnext5 * idnext0
        trans_a5i1 = aggnext5 * idnext1 
        
        ### compute relative Euler equation error
        # compute expectations
        mu = (
            trans_a0i0 * Vnext_a0_i0 ** (1. - sigma)
            + trans_a0i1 * Vnext_a0_i1 ** (1. - sigma)
            + trans_a1i0 * Vnext_a1_i0 ** (1. - sigma)
            + trans_a1i1 * Vnext_a1_i1 ** (1. - sigma)
            + trans_a2i0 * Vnext_a2_i0 ** (1. - sigma)
            + trans_a2i1 * Vnext_a2_i1 ** (1. - sigma)
            + trans_a3i0 * Vnext_a3_i0 ** (1. - sigma)
            + trans_a3i1 * Vnext_a3_i1 ** (1. - sigma)
            + trans_a4i0 * Vnext_a4_i0 ** (1. - sigma)
            + trans_a4i1 * Vnext_a4_i1 ** (1. - sigma)
            + trans_a5i0 * Vnext_a5_i0 ** (1. - sigma)
            + trans_a5i1 * Vnext_a5_i1 ** (1. - sigma)
            ) ** (1. / (1. - sigma))
        
        exp = (
            trans_a0i0 * (cnext_a0_i0) ** (- rho) * (Vnext_a0_i0 / mu) ** (rho - sigma)
            + trans_a0i1 * (cnext_a0_i1) ** (- rho) * (Vnext_a0_i1 / mu) ** (rho - sigma)
            + trans_a1i0 * (cnext_a1_i0) ** (- rho) * (Vnext_a1_i0 / mu) ** (rho - sigma)
            + trans_a1i1 * (cnext_a1_i1) ** (- rho) * (Vnext_a1_i1 / mu) ** (rho - sigma)
            + trans_a2i0 * (cnext_a2_i0) ** (- rho) * (Vnext_a2_i0 / mu) ** (rho - sigma)
            + trans_a2i1 * (cnext_a2_i1) ** (- rho) * (Vnext_a2_i1 / mu) ** (rho - sigma)
            + trans_a3i0 * (cnext_a3_i0) ** (- rho) * (Vnext_a3_i0 / mu) ** (rho - sigma)
            + trans_a3i1 * (cnext_a3_i1) ** (- rho) * (Vnext_a3_i1 / mu) ** (rho - sigma)
            + trans_a4i0 * (cnext_a4_i0) ** (- rho) * (Vnext_a4_i0 / mu) ** (rho - sigma)
            + trans_a4i1 * (cnext_a4_i1) ** (- rho) * (Vnext_a4_i1 / mu) ** (rho - sigma)
            + trans_a5i0 * (cnext_a5_i0) ** (- rho) * (Vnext_a5_i0 / mu) ** (rho - sigma)
            + trans_a5i1 * (cnext_a5_i1) ** (- rho) * (Vnext_a5_i1 / mu) ** (rho - sigma)        
        )
        
        # compute relee
        releue = ((beta * exp + lambd) / p)** (- 1. / rho) / c - 1.
        
        Vopt = ((1. - beta) * c ** (1. - rho) + beta * mu ** (1. - rho)) ** (1. / (1. - rho))
        relbee = (Vopt / V) - 1.
        
        ### compute KKT condition error
        KKTerr = anext * lambd
        
        ### compute punishment
        punish = (tf.math.abs(c_try - c)
                + tf.math.abs(cnext_a0_i0 - cnext_a0_i0_try)
                + tf.math.abs(cnext_a0_i1 - cnext_a0_i1_try)
                + tf.math.abs(cnext_a1_i0 - cnext_a1_i0_try)
                + tf.math.abs(cnext_a1_i1 - cnext_a1_i1_try)
                + tf.math.abs(cnext_a2_i0 - cnext_a2_i0_try)
                + tf.math.abs(cnext_a2_i1 - cnext_a2_i1_try)
                + tf.math.abs(cnext_a3_i0 - cnext_a3_i0_try)
                + tf.math.abs(cnext_a3_i1 - cnext_a3_i1_try)              
                + tf.math.abs(cnext_a4_i0 - cnext_a4_i0_try)
                + tf.math.abs(cnext_a4_i1 - cnext_a4_i1_try)
                + tf.math.abs(cnext_a5_i0 - cnext_a5_i0_try)
                + tf.math.abs(cnext_a5_i1 - cnext_a5_i1_try)                
                ) * 100
        
        # Market clearing
        # This may be an inefficient computation as we are computing weights_next, which only depends on X_agg, for 
        # multiple of the same X_agg realizations. 
        weights_next = get_weights_eta(net_pol, X_agg)

        aggdem = tf.math.reduce_sum(weights_next * tf.tile(tf.reshape(agrid, [1, nA]), [1, 2]), axis=1, keepdims=True)
        
        mcerr = tf.math.abs(aggdem - 1.) * nZ

        error = tf.concat([releue, relbee, mcerr, KKTerr, punish], axis=1)
        
        cost = tf.math.reduce_sum(error ** 2) / nSamples
    
        return cost, c, releue, relbee, exp, lambd, mu, mcerr, KKTerr

    #=========================================================================================================================================
    # Simulation
    #=========================================================================================================================================
    def sim_shocks(z0idx, T):
        ntracks = z0idx.shape[0]
        rands = np.random.rand(ntracks, T)
        zidx_evol = np.empty((ntracks, T), np.int32)
        zidx_evol[:, 0:1] = z0idx
        
        for i in range(ntracks):
            for t in range(1, T):
                zoldidx = zidx_evol[i, t - 1]

                rand = rands[i, t- 1]
                agg_trans = 0.
                for znidx in range(nZ):
                    agg_trans += pi_Z_np[int(zoldidx), int(znidx)]
                    if rand <= agg_trans:
                        zidx_evol[i, t] = znidx
                        zoldidx = znidx
                        break

        return zidx_evol

    # Note: this function is currently still not optimized
    def get_sequence_agg(zidx_evol, Xagg_0, net):
        nAgg = Xagg_0.shape[0]
        nAgg2 = zidx_evol.shape[0]
        
        assert nAgg2 == nAgg, 'same number of starting points for zidx_evol and Xagg_0 should be given'
        
        T = zidx_evol.shape[1]
        
        Xagg_evol = []
        Xaggnext0_evol = []
        Xaggnext1_evol = []
        Xaggnext2_evol = []
        Xaggnext3_evol = []    
        Xaggnext4_evol = []
        Xaggnext5_evol = []       
        
        Xagg_evol.append(Xagg_0)
        
        
        for t in range(T):
            Xagg = Xagg_evol[-1]
            Xaggnext0, Xaggnext1, Xaggnext2, Xaggnext3, Xaggnext4, Xaggnext5 = get_Xagg_next(net, Xagg)
            Xaggnext0_evol.append(Xaggnext0)
            Xaggnext1_evol.append(Xaggnext1)
            Xaggnext2_evol.append(Xaggnext2)
            Xaggnext3_evol.append(Xaggnext3)    
            Xaggnext4_evol.append(Xaggnext4)
            Xaggnext5_evol.append(Xaggnext5)    
            
            if t < T - 1:
                zidxnext = zidx_evol[:, t+1]
                mask0 = tf.reshape(tf.cast(tf.cast(zidxnext, tf.float32) < 0.5, tf.float32), [-1, 1])
                mask1 = tf.reshape(tf.cast(tf.cast(zidxnext, tf.float32) < 1.5, tf.float32), [-1, 1]) - mask0
                mask2 = tf.reshape(tf.cast(tf.cast(zidxnext, tf.float32) < 2.5, tf.float32), [-1, 1]) - mask1 - mask0
                mask3 = tf.reshape(tf.cast(tf.cast(zidxnext, tf.float32) < 3.5, tf.float32), [-1, 1]) - mask2 - mask1 - mask0
                mask4 = tf.reshape(tf.cast(tf.cast(zidxnext, tf.float32) < 4.5, tf.float32), [-1, 1]) - mask3 - mask2 - mask1 - mask0
                mask5 = tf.reshape(tf.cast(tf.cast(zidxnext, tf.float32) < 5.5, tf.float32), [-1, 1]) - mask4 - mask3 - mask2 - mask1 - mask0

                Xagg_new = mask0 * Xaggnext0 + mask1 * Xaggnext1 + mask2 * Xaggnext2 + mask3 * Xaggnext3 + mask4 * Xaggnext4 + mask5 * Xaggnext5
                Xagg_evol.append(Xagg_new)
                
        Xagg_evol = tf.reshape(tf.concat(Xagg_evol, axis=0), [T * nAgg, -1])
        Xaggnext0_evol = tf.reshape(tf.concat(Xaggnext0_evol, axis=0), [T * nAgg, -1])
        Xaggnext1_evol = tf.reshape(tf.concat(Xaggnext1_evol, axis=0), [T * nAgg, -1])
        Xaggnext2_evol = tf.reshape(tf.concat(Xaggnext2_evol, axis=0), [T * nAgg, -1])
        Xaggnext3_evol = tf.reshape(tf.concat(Xaggnext3_evol, axis=0), [T * nAgg, -1])
        Xaggnext4_evol = tf.reshape(tf.concat(Xaggnext4_evol, axis=0), [T * nAgg, -1])
        Xaggnext5_evol = tf.reshape(tf.concat(Xaggnext5_evol, axis=0), [T * nAgg, -1])
        
        return Xagg_evol, Xaggnext0_evol, Xaggnext1_evol, Xaggnext2_evol, Xaggnext3_evol, Xaggnext4_evol, Xaggnext5_evol


    def create_training_data(N_agg, N_id_per_shock, Xagg_0, net):
        ntracks = Xagg_0.shape[0]
        z0idx = Xagg_0[:, 0 : 1]

        nTagg = int(N_agg / ntracks)
        
        if not((N_agg / ntracks) == nTagg):
            print('Number of aggregate samples not divisible by number of tracks')
        
        
        # get sequence of aggregate exogenous shocks
        zidx_evol = sim_shocks(z0idx, nTagg)
        
        # get sequence of aggregate states
        Xagg_evol, Xaggnext0_evol, Xaggnext1_evol, Xaggnext2_evol, Xaggnext3_evol, Xaggnext4_evol, Xaggnext5_evol = get_sequence_agg(tf.cast(zidx_evol, tf.float32), Xagg_0, net)
        
        # copy the aggregate states
        Xagg_evol = tf.tile(Xagg_evol, [nEta * N_id_per_shock, 1])
        Xaggnext0_data = tf.tile(Xaggnext0_evol, [nEta * N_id_per_shock, 1])
        Xaggnext1_data = tf.tile(Xaggnext1_evol, [nEta * N_id_per_shock, 1])
        Xaggnext2_data = tf.tile(Xaggnext2_evol, [nEta * N_id_per_shock, 1])
        Xaggnext3_data = tf.tile(Xaggnext3_evol, [nEta * N_id_per_shock, 1])
        Xaggnext4_data = tf.tile(Xaggnext4_evol, [nEta * N_id_per_shock, 1])
        Xaggnext5_data = tf.tile(Xaggnext5_evol, [nEta * N_id_per_shock, 1])
        
        # create idiosyncratic shocks
        a = tf.random.uniform((nEta * N_id_per_shock * ntracks * nTagg, 1)) * (amax - amin) + amin
        etaidx = tf.cast(tf.random.uniform((nEta * N_id_per_shock * ntracks * nTagg, 1)) < 0.5, tf.float32)
        
        X_data = tf.concat([etaidx, a, Xagg_evol], axis=1)
        
        # construct training data
        return X_data, Xaggnext0_data, Xaggnext1_data, Xaggnext2_data, Xaggnext3_data, Xaggnext4_data, Xaggnext5_data, Xagg_evol

    # for plotting
    def create_plotting_data(N_agg, N_id_plot, Xagg_0, net):
        ntracks = Xagg_0.shape[0]
        z0idx = Xagg_0[:, 0 : 1]

        nTagg = int(N_agg / ntracks)
        
        if not((N_agg / ntracks) == nTagg):
            print('Number of aggregate samples not divisible by number of tracks')
        
        
        # get sequence of aggregate exogenous shocks
        zidx_evol = sim_shocks(z0idx, nTagg)
        
        # get sequence of aggregate states
        Xagg_evol, Xaggnext0_evol, Xaggnext1_evol, Xaggnext2_evol, Xaggnext3_evol, Xaggnext4_evol, Xaggnext5_evol = get_sequence_agg(tf.cast(zidx_evol, tf.float32), Xagg_0, net)
        
        # copy the aggregate states
        Xagg_evol = tf.tile(Xagg_evol, [nEta * N_id_plot, 1])
        Xaggnext0_data = tf.tile(Xaggnext0_evol, [nEta * N_id_plot, 1])
        Xaggnext1_data = tf.tile(Xaggnext1_evol, [nEta * N_id_plot, 1])
        Xaggnext2_data = tf.tile(Xaggnext2_evol, [nEta * N_id_plot, 1])
        Xaggnext3_data = tf.tile(Xaggnext3_evol, [nEta * N_id_plot, 1])
        Xaggnext4_data = tf.tile(Xaggnext4_evol, [nEta * N_id_plot, 1])
        Xaggnext5_data = tf.tile(Xaggnext5_evol, [nEta * N_id_plot, 1])
        
        # create idiosyncratic shocks
        aplotgrid = tf.linspace(amin, amax, N_id_plot)
        aplot = tf.reshape(tf.repeat(aplotgrid, nEta * N_agg), [-1, 1])

        etaidx = tf.tile(tf.concat([tf.zeros([N_agg, 1]), tf.ones([N_agg, 1])], axis = 0), [N_id_plot, 1])
        
        X_data = tf.concat([etaidx, aplot, Xagg_evol], axis=1)
        
        # construct training data
        return X_data, Xaggnext0_data, Xaggnext1_data, Xaggnext2_data, Xaggnext3_data, Xaggnext4_data, Xaggnext5_data, Xagg_evol, aplotgrid


    def create_tfdata(training_data_X, training_data_Xaggnext0, training_data_Xaggnext1, training_data_Xaggnext2, training_data_Xaggnext3, training_data_Xaggnext4, training_data_Xaggnext5, buffer_size, batch_size):
        train_dataset = tf.data.Dataset.from_tensor_slices((training_data_X, training_data_Xaggnext0, training_data_Xaggnext1, training_data_Xaggnext2, training_data_Xaggnext3, training_data_Xaggnext4, training_data_Xaggnext5))
        train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size)
        return train_dataset


    #=========================================================================================================================================
    # Set up optimizer
    #=========================================================================================================================================
    if optimizer_name == 'adam':
        optimizer_joint = keras.optimizers.Adam(learning_rate=lr)
    else:
        raise NotImplementedError

    #=========================================================================================================================================
    # Set up price network
    #=========================================================================================================================================
    num_hidden_layers_price = len(num_hidden_nodes_price)
    
    inputs_price = keras.Input(shape=(nA * nEta + 3,))
    for layerindex in range(num_hidden_layers_price):
        num_nodes = num_hidden_nodes_price[layerindex]
        activation = activations_hidden_nodes_price[layerindex]
        if layerindex == 0:
            x = keras.layers.Dense(num_nodes, activation=activation)(inputs_price)
        else:
            x = keras.layers.Dense(num_nodes, activation=activation)(x)
    
    outputs_price = keras.layers.Dense(1, activation="softplus")(x)

    net_price = keras.Model(inputs=inputs_price, outputs=outputs_price)
    net_price.build(input_shape=(None, 3 + nEta * nA))

    #=========================================================================================================================================
    # Set up policy network
    #=========================================================================================================================================
    num_hidden_layers_pol = len(num_hidden_nodes_pol)
    
    inputs_pol = keras.Input(shape=(nA * nEta + 5,))
    for layerindex in range(num_hidden_layers_pol):
        num_nodes = num_hidden_nodes_pol[layerindex]
        activation = activations_hidden_nodes_pol[layerindex]
        if layerindex == 0:
            x = keras.layers.Dense(num_nodes, activation=activation)(inputs_pol)
        else:
            x = keras.layers.Dense(num_nodes, activation=activation)(x)
    
    outputs_pol = keras.layers.Dense(3, activation="softplus")(x)
    
    net_pol = keras.Model(inputs=inputs_pol, outputs=outputs_pol)
    net_pol.build(input_shape=(None, 5 + nEta * nA))

    #=========================================================================================================================================
    # Define training step
    #=========================================================================================================================================
    joined_parameters = net_pol.trainable_weights + net_price.trainable_weights


    @tf.function
    def train_step(net_pol, net_price, optimizer_joint, Xdata, Xaggnextdata0, Xaggnextdata1, Xaggnextdata2, Xaggnextdata3, Xaggnextdata4, Xaggnextdata5):
        with tf.GradientTape() as tape:
            loss_value = cost(net_pol, net_price, Xdata, Xaggnextdata0, Xaggnextdata1, Xaggnextdata2, Xaggnextdata3, Xaggnextdata4, Xaggnextdata5)[0]
        grads = tape.gradient(loss_value, joined_parameters)
        
        optimizer_joint.apply_gradients(zip(grads, joined_parameters))

        return loss_value


    #=========================================================================================================================================
    # Training
    #=========================================================================================================================================
    
    buffer_size = len_episodes * num_id_per_shock * nEta
    num_batches = buffer_size / batch_size
    
    loss = []

    last_ckpt = None

    current_episode = tf.Variable(1)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), current_episode=current_episode, 
                            optimizer_joint=optimizer_joint,
                            policy=net_pol, price=net_price)

    
    if load_bool:
        print('###################################')
        load_base_path = os.path.join(output_path, load_run_name, 'model')
        
        load_ckpt_path = os.path.join(load_base_path, load_run_name + '-episode' + str(load_episode))
        load_data_path = load_ckpt_path + '_LastData.npy'

        last_ckpt = load_ckpt_path

        ckpt.restore(last_ckpt).expect_partial()
        Xagg_0 = np.load(load_data_path)
        
        print('Loaded checkpoint and initial data from', load_ckpt_path)
        print('Loaded Xagg_0 from', load_data_path)

    if not(load_bool):
        # initialize starting points
        Xagg_0 = np.zeros([num_tracks, nEta * nA + 3])
        Xagg_0[:, 4 + 0] = 0.5
        Xagg_0[:, 4 + 3 + nA] = 0.5
        Xagg_0 = tf.constant(Xagg_0, dtype=tf.float32)

    
    for outer in range(num_episodes):
        episode_loss = 0.
        print('#' + '=' * 50)
        print('#' + '=' * 50)
        print('Episode = ', outer)
        print('#' + '=' * 50)
        # simulate data
        training_data_X, training_data_Xaggnext0, training_data_Xaggnext1, training_data_Xaggnext2, training_data_Xaggnext3, training_data_Xaggnext4, training_data_Xaggnext5, Xagg_evol  = create_training_data(len_episodes, num_id_per_shock, Xagg_0, net_pol)
        train_dataset = create_tfdata(training_data_X, training_data_Xaggnext0, training_data_Xaggnext1, training_data_Xaggnext2, training_data_Xaggnext3, training_data_Xaggnext4, training_data_Xaggnext5, buffer_size, batch_size)    
        
        # update starting points
        Xagg_0 = Xagg_evol[-num_tracks:, :].numpy()
        
        print('#' + '=' * 50)
        
        
        for epoch in range(epochs_per_episode):
            if train_flag:
                print("\nStart training: epoch %d" % (epoch,))
            else:
                print("\nEvaluating cost (no training): epoch %d" % (epoch,))
            # Iterate over the batches of the dataset.
            for step, x_batch_train in enumerate(train_dataset):
                
                Xdata = x_batch_train[0]
                Xaggnextdata0 = x_batch_train[1]
                Xaggnextdata1 = x_batch_train[2]
                Xaggnextdata2 = x_batch_train[3]
                Xaggnextdata3 = x_batch_train[4]
                Xaggnextdata4 = x_batch_train[5]
                Xaggnextdata5 = x_batch_train[6]            
                
                if train_flag:
                    batch_loss = train_step(net_pol, net_price, optimizer_joint, Xdata, Xaggnextdata0, Xaggnextdata1, Xaggnextdata2, Xaggnextdata3, Xaggnextdata4, Xaggnextdata5)
                else:
                    batch_loss =  cost(net_pol, net_price, Xdata, Xaggnextdata0, Xaggnextdata1, Xaggnextdata2, Xaggnextdata3, Xaggnextdata4, Xaggnextdata5)[0]
                if epoch == 0:
                    episode_loss += batch_loss / num_batches
                    

            if (epoch % 100 == 0) and (outer % save_interval == 0):
                current_episode.assign_add(1)
                save_nm = run_name + '_episode{}'.format(outer + load_episode)
                output_ckpt = ckpt.save(os.path.join(save_path, save_nm))
                print('Checkpoint saved to', output_ckpt)
                
                np.save(os.path.join(save_path, save_nm + '_LastData.npy'), Xagg_0)
                print('Aggregate starting state saved to', os.path.join(save_path, save_nm + '_LastData.npy'))


            if epoch == 0:
                print('Episode {}, epoch {}, loss [log10] = {:.3f}'.format(outer, epoch, np.log10(episode_loss.numpy())))
                loss.append(episode_loss.numpy())

        if outer % save_interval == 0:
            plot_name = os.path.join(plot_path, 'episode{}_'.format(outer + load_episode))
            
            _, c, releue, relbee, _, _, _, mcerr, _ = cost(net_pol, net_price,  Xdata, Xaggnextdata0, Xaggnextdata1, Xaggnextdata2, Xaggnextdata3, Xaggnextdata4, Xaggnextdata5)

            mask_id_low = Xdata[:, 0].numpy() == 0
            mask_id_high = Xdata[:, 0].numpy() == 1
            
            plt.figure(figsize=std_figsize)
            plt.plot(Xdata.numpy()[mask_id_low, 1], c.numpy()[mask_id_low], 'kx', label = r'$\eta_t=0.8$')
            plt.plot(Xdata.numpy()[mask_id_high, 1], c.numpy()[mask_id_high], 'rx', label = r'$\eta_t=1.2$')
            plt.legend()
            plt.xlabel('Asset holding')
            plt.ylabel('c')
            plt.legend()
            plt.savefig(plot_name + 'c.pdf', bbox_inches='tight')
            plt.close()        
            
            plt.figure(figsize=std_figsize)
            plt.plot(Xdata[:, 1].numpy(), relbee.numpy(), 'rx', label = 'Bellman')
            plt.plot(Xdata[:, 1].numpy(), releue.numpy(), 'kx', label = 'Euler')
            plt.xlabel('Asset holding')
            plt.ylabel('Rel. errors')
            plt.legend()
            plt.savefig(plot_name + 'relerr.pdf', bbox_inches='tight')
            plt.close()

            plt.figure(figsize=std_figsize)
            plt.hist(100 * np.abs(mcerr.numpy().flatten() / nZ), color = 'k', alpha = 0.5)
            plt.xlabel(r'Market clearing error [$10^{-2}$]')
            plt.ylabel('Count')
            plt.savefig(plot_name + 'mcerr.pdf', bbox_inches = 'tight')
            plt.close()

            plt.figure(figsize=std_figsize)
            plt.plot(Xdata.numpy()[mask_id_low, 1], net_pol(Xdata).numpy()[mask_id_low, 0], 'kx', label = r'$\eta_t=0.8$')
            plt.plot(Xdata.numpy()[mask_id_high, 1], net_pol(Xdata).numpy()[mask_id_high, 0], 'rx', label = r'$\eta_t=1.2$')
            plt.xlabel('Asset holding')
            plt.ylabel('Savings')
            plt.legend()
            plt.savefig(plot_name + 'saving.pdf', bbox_inches='tight')
            plt.close()        

            plt.figure(figsize=std_figsize)
            plt.plot(Xdata.numpy()[mask_id_low, 1], net_pol(Xdata).numpy()[mask_id_low, 1], 'kx', label = r'$\eta_t=0.8$')
            plt.plot(Xdata.numpy()[mask_id_high, 1], net_pol(Xdata).numpy()[mask_id_high, 1], 'rx', label = r'$\eta_t=1.2$')
            plt.xlabel('Asset holding')
            plt.ylabel('Mutiplier')
            plt.legend()
            plt.savefig(plot_name + 'mult.pdf', bbox_inches='tight')
            plt.close()        

            plt.figure(figsize=std_figsize)
            plt.plot(Xdata.numpy()[mask_id_low, 1], net_pol(Xdata).numpy()[mask_id_low, 2], 'kx', label = r'$\eta_t=0.8$')
            plt.plot(Xdata.numpy()[mask_id_high, 1], net_pol(Xdata).numpy()[mask_id_high, 2], 'rx', label = r'$\eta_t=1.2$')
            plt.xlabel('Asset holding')
            plt.ylabel('Value function')
            plt.legend()
            plt.savefig(plot_name + 'valuefunction.pdf', bbox_inches='tight')
            plt.close()        
            
            plt.figure(figsize=std_figsize)
            plt.plot(np.log10(loss), 'k-')
            plt.ylabel('loss [log10]')
            plt.savefig(plot_name + 'loss.pdf', bbox_inches='tight')
            plt.close()      
  

            plt.figure(figsize=std_figsize)
            cols = ['r', 'b', 'y', 'k', 'g', 'm']
            for i in range(nZ):
                plt.hist(net_price(Xagg_evol[Xagg_evol[:, 0] == i])[:, 0].numpy(), color = cols[i], alpha = 0.4,  label='shock = '+str(i))
            plt.legend()
            plt.xlabel('Bond price')
            plt.ylabel('Count')
            plt.savefig(plot_name + 'pricehist.pdf', bbox_inches='tight')
            plt.close()        


            plt.figure(figsize=std_figsize)
            for i in range(nZ):
                plt.plot(Xdata.numpy()[(Xdata[:, 2] == i).numpy(), 1], c[(Xdata[:, 2] == i).numpy()], cols[i]+'x', label='agg. shock = '+str(i))
            plt.legend()
            plt.ylabel('c')
            plt.xlabel('Asset holding')
            plt.savefig(plot_name + 'c_by_aggshocks.pdf', bbox_inches='tight')
            plt.close()     

            plt.figure(figsize=std_figsize)
            plt.plot(agrid, Xagg_evol[-1, 3 : 3 + nA], 'k-', label = r'$\eta_t=0.8$')
            plt.plot(agrid, Xagg_evol[-1, 3 + nA : 3 + 2 * nA],  'r-', label = r'$\eta_t=1.2$')
            plt.xlabel('Asset holding')
            plt.ylabel('Distribution')
            plt.legend()
            plt.savefig(plot_name + 'distributions.pdf', bbox_inches='tight')
            plt.close()        

            #=========================================================================================================================================
            # For plotting more comprehensive statistics of errors
            #=========================================================================================================================================
            
            if print_more_error_stats:
                if outer % print_more_stats_interval == 0:
                    print('#' + '=' * 50)
                    print('generating more detailed error statistics')
                    print('#' + '=' * 50)
                    
                    N_id_plot = 50
                    Xagg_0_plot = Xagg_evol[-num_tracks:, :].numpy()

                    plt_X, plt_Xaggnext0, plt_Xaggnext1, plt_Xaggnext2, plt_Xaggnext3, plt_Xaggnext4, plt_Xaggnext5, plt_Xagg_evol, plt_aplotgrid  = create_plotting_data(len_episodes, N_id_plot, Xagg_0_plot, net_pol)

                    ntotplot = plt_X.shape[0]

                    c_all = np.empty(ntotplot)
                    releue_all = np.empty(ntotplot)
                    relbee_all = np.empty(ntotplot)
                    exp_all = np.empty(ntotplot)
                    lambd_all = np.empty(ntotplot)
                    mu_all = np.empty(ntotplot)
                    mcerr_all = np.empty(ntotplot)
                    KKTerr_all = np.empty(ntotplot)

                    npackets = 50
                    assert ntotplot % npackets == 0, 'make sure its divisible'

                    n_per_packet = ntotplot / npackets

                    for pcounter in range(npackets):
                        idx_low = int(pcounter * n_per_packet)
                        idx_high = int(idx_low + n_per_packet)                 
                    
                        plt_cost, plt_c, plt_releue, plt_relbee, plt_exp, plt_lambd, plt_mu, plt_mcerr, plt_KKTerr = cost(
                            net_pol, net_price, 
                            plt_X[idx_low : idx_high, :], 
                            plt_Xaggnext0[idx_low : idx_high, :], plt_Xaggnext1[idx_low : idx_high, :], plt_Xaggnext2[idx_low : idx_high, :], 
                            plt_Xaggnext3[idx_low : idx_high, :], plt_Xaggnext4[idx_low : idx_high, :], plt_Xaggnext5[idx_low : idx_high, :])


                        c_all[idx_low : idx_high] = plt_c[:, 0]
                        releue_all[idx_low : idx_high] = plt_releue[:, 0]
                        relbee_all[idx_low : idx_high] = plt_relbee[:, 0]
                        exp_all[idx_low : idx_high] = plt_exp[:, 0]
                        lambd_all[idx_low : idx_high] = plt_lambd[:, 0]
                        mu_all[idx_low : idx_high] = plt_mu[:, 0]
                        mcerr_all[idx_low : idx_high] = plt_mcerr[:, 0]
                        KKTerr_all[idx_low : idx_high] = plt_KKTerr[:, 0]            

                    # reshape into usable format
                    c_use = tf.reshape(c_all, [N_id_plot, 2, len_episodes])
                    releue_use = tf.reshape(releue_all, [N_id_plot, 2, len_episodes])
                    relbee_use = tf.reshape(relbee_all, [N_id_plot, 2, len_episodes])
                    exp_use = tf.reshape(exp_all, [N_id_plot, 2, len_episodes])
                    lambd_use = tf.reshape(lambd_all, [N_id_plot, 2, len_episodes])
                    mu_use = tf.reshape(mu_all, [N_id_plot, 2, len_episodes])
                    mcerr_use = tf.reshape(mcerr_all, [N_id_plot, 2, len_episodes])
                    KKTerr_use = tf.reshape(KKTerr_all, [N_id_plot, 2, len_episodes])


                    plt.figure(figsize=std_figsize)
                    ax1 = plt.subplot(1,1,1)
                    ax1.plot(plt_aplotgrid, np.log10(np.mean(np.abs(releue_use[:, 0, :]), axis = 1)), 'r-', label = r'$\eta_t=0.8$, mean')
                    ax1.plot(plt_aplotgrid, np.log10(np.mean(np.abs(releue_use[:, 1, :]), axis = 1)), 'k-', label = r'$\eta_t=1.2$, mean')
                    
                    for perc in perc_list:
                        ax1.plot(plt_aplotgrid, np.log10(np.percentile(np.abs(releue_use[:, 0, :]), perc, axis = 1)), 'r' + perc_ls[perc], label = r'$\eta_t=0.8$, '+str(perc)+' perc.' )
                        ax1.plot(plt_aplotgrid, np.log10(np.percentile(np.abs(releue_use[:, 1, :]), perc, axis = 1)), 'k' + perc_ls[perc], label = r'$\eta_t=1.2$, '+str(perc)+' perc.' )
                        
                    ax1.set_xlabel('Asset holding')
                    ax1.set_ylabel('Rel. Ee error [log10]')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.savefig(plot_name + 'detailed_relee.pdf', bbox_inches='tight')
                    plt.close()   


                    plt.figure(figsize=std_figsize)
                    ax2 = plt.subplot(1,1,1)
                    ax2.plot(plt_aplotgrid, np.log10(np.mean(np.abs(relbee_use[:, 0, :]), axis = 1)), 'r-', label = r'$\eta_t=0.8$, mean')
                    ax2.plot(plt_aplotgrid, np.log10(np.mean(np.abs(relbee_use[:, 1, :]), axis = 1)), 'k-', label = r'$\eta_t=1.2$, mean')

                    for perc in perc_list:
                        ax2.plot(plt_aplotgrid, np.log10(np.percentile(np.abs(relbee_use[:, 0, :]), perc, axis = 1)), 'r' + perc_ls[perc], label = r'$\eta_t=0.8$, '+str(perc)+' perc.')
                        ax2.plot(plt_aplotgrid, np.log10(np.percentile(np.abs(relbee_use[:, 1, :]), perc, axis = 1)), 'k' + perc_ls[perc], label = r'$\eta_t=1.2$, '+str(perc)+' perc.')

                    ax2.set_xlabel('Asset holding')
                    ax2.set_ylabel('Rel. Be error [log10]')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.savefig(plot_name + 'detailed_relbe.pdf', bbox_inches='tight')
                    plt.close()        

                    plt.figure(figsize=std_figsize)
                    ax2 = plt.subplot(1,1,1)
                    ax2.plot(plt_aplotgrid, np.log10(np.mean(np.abs(KKTerr_use[:, 0, :]), axis = 1)), 'r-', label = r'$\eta_t=0.8$, mean')
                    ax2.plot(plt_aplotgrid, np.log10(np.mean(np.abs(KKTerr_use[:, 1, :]), axis = 1)), 'k-', label = r'$\eta_t=1.2$, mean')

                    for perc in perc_list:
                        ax2.plot(plt_aplotgrid, np.log10(np.percentile(np.abs(KKTerr_use[:, 0, :]), perc, axis = 1)), 'r' + perc_ls[perc], label = r'$\eta_t=0.8$, '+str(perc)+' perc.')
                        ax2.plot(plt_aplotgrid, np.log10(np.percentile(np.abs(KKTerr_use[:, 1, :]), perc, axis = 1)), 'k' + perc_ls[perc], label = r'$\eta_t=1.2$, '+str(perc)+' perc.')

                    ax2.set_xlabel('Asset holding')
                    ax2.set_ylabel('KKT error [log10]')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.savefig(plot_name + 'detailed_KKTerr.pdf', bbox_inches='tight')
                    plt.close()        



                    plt.figure(figsize=std_figsize)
                    ax3 = plt.subplot(1,1,1)
                    ax3.plot(plt_aplotgrid, np.mean(c_use[:, 0, :], axis = 1), 'k-', label = r'$\eta_t=0.8$')
                    ax3.plot(plt_aplotgrid, np.mean(c_use[:, 1, :], axis = 1), 'k--', label = r'$\eta_t=1.2$')
                    ax3.set_xlabel('Asset holding')
                    ax3.set_ylabel('c')
                    plt.legend()
                    plt.savefig(plot_name + 'detailed_c.pdf', bbox_inches='tight')
                    plt.close()        

                    
                    plt.figure(figsize=std_figsize)
                    ax4 = plt.subplot(1,1,1)
                    ax4.hist(100 * (mcerr.numpy().flatten() / nZ), color = 'k', alpha = 0.5)
                    ax4.set_xlabel(r'Error in market clearing [$10^{-2}$]')
                    ax4.set_ylabel('Count')
                    #plt.legend()
                    plt.savefig(plot_name + 'detailed_marketclearing.pdf', bbox_inches='tight')
                    plt.close() 
                                                        
                    

                    plt.figure(figsize=std_figsize)
                    ax5 = plt.subplot(1,1,1)
                    ax5.plot(plt_aplotgrid, np.mean(lambd_use[:, 0, :], axis = 1), 'r-', label = r'$\eta_t=0.8$, mean')
                    ax5.plot(plt_aplotgrid, np.mean(lambd_use[:, 1, :], axis = 1), 'k-', label = r'$\eta_t=1.2$, mean')

                    for perc in perc_list:
                        ax5.plot(plt_aplotgrid, np.percentile(lambd_use[:, 0, :], perc, axis = 1), 'r' + perc_ls[perc], label = r'$\eta_t=0.8$, '+str(perc)+' perc.')
                        ax5.plot(plt_aplotgrid, np.percentile(lambd_use[:, 1, :], perc, axis = 1), 'k' + perc_ls[perc], label = r'$\eta_t=1.2$, '+str(perc)+' perc.')

                    ax5.set_xlabel('Asset holding')
                    ax5.set_ylabel('KKT multiplier')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.savefig(plot_name + 'detailed_KKTmult.pdf', bbox_inches='tight')
                    plt.close()    

                    print('#' + '=' * 50)
                    print('more detailed error statistics done')
                    print('#' + '=' * 50)

def main():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_from_scratch', dest='load_flag', action='store_false')
    parser.set_defaults(load_flag=True)
    args = parser.parse_args()
    load_flag = args.load_flag

    print('##### input arguments #####')
    seed = 1
    path_wd = '.'
    run_name = 'deqn_continuumagents_restart' if args.load_flag else 'deqn_continuumagents'
    num_hidden_nodes_price = [500, 500]
    activations_hidden_nodes_price = [tf.nn.relu, tf.nn.relu]
    num_hidden_nodes_pol = [500, 500]
    activations_hidden_nodes_pol = [tf.nn.relu, tf.nn.relu]
    optimizer = 'adam'
    batch_size = 128
    num_episodes = 65000
    len_episodes = 2048
    epochs_per_episode = 1
    num_tracks = 16
    num_id_per_shock = 2#20
    save_interval = 50
    lr = 1e-5
    load_run_name = 'deqn_continuumagents_final' if args.load_flag else None
    load_episode = 75000 if args.load_flag else 0

    print('seed: {}'.format(seed))
    print('working directory: ' + path_wd)
    print('run_name: {}'.format(run_name))
    print('hidden nodes for price network: [500, 500]')
    print('activation hidden nodes for price network: [relu, relu]')
    print('hidden nodes for policy network: [500, 500]')
    print('activation hidden nodes for policy network: [relu, relu]')
    
    if args.load_flag:
        train_flag = False
        num_episodes = 1
        print('loading weights from deqn_continuumagents_final')
        print('loading from episode {}'.format(load_episode))
    else:
        train_flag = True
        print('optimizer: {}'.format(optimizer))
        print('batch_size: {}'.format(batch_size))
        print('num_episodes: {}'.format(num_episodes))
        print('len_episodes: {}'.format(len_episodes))
        print('epochs_per_episode: {}'.format(epochs_per_episode))
        print('num_tracks: {}'.format(num_tracks))
        print('num_id_per_shock: {}'.format(num_id_per_shock))
        print('save_interval: {}'.format(save_interval))
        print('lr: {}'.format(lr))

    print('###########################')   
    
    train(
        seed, lr, optimizer, 
        num_hidden_nodes_price, activations_hidden_nodes_price, 
        num_hidden_nodes_pol, activations_hidden_nodes_pol, 
        batch_size, num_episodes, len_episodes, epochs_per_episode, num_tracks, num_id_per_shock,
        path_wd, run_name,
        save_interval,
        train_flag, load_flag, load_run_name, load_episode      
    )


if __name__ == '__main__':
    main()
