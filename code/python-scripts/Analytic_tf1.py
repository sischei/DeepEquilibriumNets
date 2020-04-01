"""A simple Deep Equilibrium Net (DEQN) implementation.

In this notebook, we use `TensorFlow 1.13.1` to showcase the general workflow of setting
up and solving dynamic general equilibrium models with deep equilibrium nets. Use

`pip install tensorflow==1.13.1`

to install the correct version.

The notebook accompanies the working paper by Azinovic, Gaegauf, & Scheidegger(2020)
(see https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3393482) and corresponds to the
model solved in appendix C with a slighly different model calibration.

Note that this is a minimal working example for illustration purposes only. A more
extensive implementation will be published on Github.

The economic model we are solving in this example is one for which we know an exact
solution. We chose to solve the model from Krueger and Kubler (2004)
(see https://www.sciencedirect.com/science/article/pii/S0165188903001118) because, in
addition to being analytically solvable, it is closely related to the models solved in the
paper. Therefore, the approach presented in this notebook translates directly to the models
in working paper.


To change the economic parameters and neural network hyper-parameters, see ./param.py.
The load episode, seed, and plot and save intervals can be passed in via the command line:

> python Analytic_tf1.py --seed 0 --load_episode 0 --plot_interval 20 --save_interval 100

The distribution of capital, neural network loss, and the relative error in the Euler
equations are plotted every `plot_interval` steps and saved to './output/plots'. Additionally,
since we have the analytical solution, we can compare the DEQNs performance relative to the 
true solution. Therefore, we also plot the each agents policy function.
"""
# ########################################################################### #
#                              Set up workspace                               #
# ########################################################################### #
# Import modules
import os
import argparse
from datetime import datetime

import tensorflow as tf
import numpy as np
from utils import initialize_nn_weight, random_mini_batches
from params import *

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')
std_figsize = (4, 4)

# Make sure that we are working with tensorflow 1
print('tf version:', tf.__version__)
assert tf.__version__[0] == '1'

parser = argparse.ArgumentParser(description='A simple Deep Equilibrium Net implementation.')
parser.add_argument('--load_episode', type=int, default=0, help='Episode to load weights and starting point from.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--plot_interval', type=int, default=20, help='Interval to plot results.')
parser.add_argument('--save_interval', type=int, default=100, help='Interval to save model.')

# ########################################################################### #
#                           Economic util functions                           #
# ########################################################################### #

def firm(K, eta, alpha, delta):
    """Calculate return, wage and aggregate production.
    
    r = eta * K^(alpha-1) * L^(1-alpha) + (1-delta)
    w = eta * K^(alpha) * L^(-alpha)
    Y = eta * K^(alpha) * L^(1-alpha) + (1-delta) * K 

    Args:
        K: aggregate capital,
        eta: TFP value,
        alpha: output elasticity,
        delta: depreciation value.

    Returns:
        return: return (marginal product of capital), 
        wage: wage (marginal product of labor),
        Y: aggregate production.
    """
    L = tf.ones_like(K)

    r = alpha * eta * K**(alpha - 1) * L**(1 - alpha) + (1 - delta)
    w = (1 - alpha) * eta * K**alpha * L**(-alpha)
    Y = eta * K**alpha * L**(1 - alpha) + (1 - delta) * K

    return r, w, Y

def shocks(z, eta, delta):
    """Calculates tfp and depreciation based on current exogenous shock.

    Args:
        z: current exogenous shock (in {1, 2, 3, 4}),
        eta: tensor of TFP values to sample from,
        delta: tensor of depreciation values to sample from.

    Returns:
        tfp: TFP value of exogenous shock, 
        depreciation: depreciation values of exogenous shock.
    """
    tfp = tf.gather(eta, tf.cast(z, tf.int32))
    depreciation = tf.gather(delta, tf.cast(z, tf.int32))
    return tfp, depreciation
    
def wealth(k, R, l, W):
    """Calculates the wealth of the agents.

    Args:
        k: capital distribution,
        R: matrix of return,
        l: labor distribution,
        W: matrix of wages.

    Returns:
        fin_wealth: financial wealth distribution,
        lab_wealth: labor income distribution,
        tot_income: total income distribution.
    """
    fin_wealth = k * R
    lab_wealth = l * W
    tot_income = tf.add(fin_wealth, lab_wealth)
    return fin_wealth, lab_wealth, tot_income

def main(args):
    """Initialize and train deep equilibrium net."""
    # Set the seed for replicable results
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # Helper variables
    eps = 0.00001  # Small epsilon value

    # ####################################################################### #
    #                             Neural network                              #
    # ####################################################################### #
    # Neural network ----------------------------------------------------------
    # We create a placeholder for X, the input data for the neural network,
    # which corresponds to the state.
    X = tf.placeholder(tf.float32, shape=(None, num_input_nodes))
    # Get number samples
    m = tf.shape(X)[0]

    # We create all of the neural network weights and biases. The weights are
    # matrices that connect the layers of the neural network. For example, W1
    # connects the input layer to the first hidden layer
    W1 = initialize_nn_weight([num_input_nodes, num_hidden_nodes[0]])
    W2 = initialize_nn_weight([num_hidden_nodes[0], num_hidden_nodes[1]])
    W3 = initialize_nn_weight([num_hidden_nodes[1], num_output_nodes])

    # The biases are extra (shift) terms that are added to each node in the
    # neural network.
    b1 = initialize_nn_weight([num_hidden_nodes[0]])
    b2 = initialize_nn_weight([num_hidden_nodes[1]])
    b3 = initialize_nn_weight([num_output_nodes])

    # Then, we create a function, to which we pass X, that generates a
    # prediction based on the current neural network weights. Note that the
    # hidden layers are ReLU activated. The output layer is not activated
    # (i.e., it is activated with the linear function).
    def nn_predict(X):
        """Generate prediction using neural network.

        Args:
            X: state: [z, k]

        """
        hidden_layer1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
        hidden_layer2 = tf.nn.relu(tf.add(tf.matmul(hidden_layer1, W2), b2))
        output_layer = tf.add(tf.matmul(hidden_layer2, W3), b3)
        return output_layer

    # ####################################################################### #
    #                             Economic model                              #
    # ####################################################################### #
    # Current period ##########################################################
    # Today's extended state: 
    z = X[:, 0]  # exogenous shock
    tfp = X[:, 1]  # total factor productivity
    depr = X[:, 2]  # depreciation
    K = X[:, 3]  # aggregate capital
    L = X[:, 4]  # aggregate labor
    r = X[:, 5]  # return on capital
    w = X[:, 6]  # wage
    Y = X[:, 7]  # aggregate production
    k = X[:, 8 : 8 + A]  # distribution of capital
    fw = X[:, 8 + A : 8 + 2 * A]  # distribution of financial wealth
    linc = X[:, 8 + 2 * A : 8 + 3 * A]  # distribution of labor income
    inc = X[:, 8 + 3 * A : 8 + 4 * A]   # distribution of total income

    # Today's assets: How much the agents save
    # Get today's assets by executing the neural network
    a = nn_predict(X)
    # The last agent consumes everything they own
    a_all = tf.concat([a, tf.zeros([m, 1])], axis=1)

    # c_orig: the original consumption predicted by the neural network However,
    #     the network can predict negative values before it learns not to. We
    #     ensure that the network learns itself out of a bad region by
    #     penalizing negative consumption. We ensure that consumption is not
    #     negative by including a penalty term on c_orig_prime_1
    # c: is the corrected version c_all_orig_prime_1, in which all negative
    #     consumption values are set to ~0. If none of the consumption values
    #     are negative then c_orig_prime_1 == c_prime_1.
    c_orig = inc - a_all
    c = tf.maximum(c_orig, tf.ones_like(c_orig) * eps)

    # Today's savings become tomorrow's capital holding, but the first agent
    # is born without a capital endowment.
    k_prime = tf.concat([tf.zeros([m, 1]), a], axis=1)

    # Tomorrow's aggregate capital
    K_prime_orig = tf.reduce_sum(k_prime, axis=1, keepdims=True)
    K_prime = tf.maximum(K_prime_orig, tf.ones_like(K_prime_orig) * eps)

    # Tomorrow's labor
    l_prime = tf.tile(labor_endow, [m, 1])
    L_prime = tf.ones_like(K_prime)

    # Next period #############################################################
    # Shock 1 -----------------------------------------------------------------
    # 1) Get remaining parts of tomorrow's extended state
    # Exogenous shock
    z_prime_1 = 0 * tf.ones_like(z)

    # TFP and depreciation
    tfp_prime_1, depr_prime_1 = shocks(z_prime_1, eta, delta)

    # Return on capital, wage and aggregate production
    r_prime_1, w_prime_1, Y_prime_1 = firm(K_prime, tfp_prime_1, alpha, depr_prime_1)
    R_prime_1 = r_prime_1 * tf.ones([1, A])
    W_prime_1 = w_prime_1 * tf.ones([1, A])

    # D istribution of financial wealth, labor income, and total income
    fw_prime_1, linc_prime_1, inc_prime_1 = wealth(k_prime, R_prime_1, l_prime, W_prime_1)

    # Tomorrow's state: Concatenate the parts together
    x_prime_1 = tf.concat([tf.expand_dims(z_prime_1, -1),
                           tfp_prime_1,
                           depr_prime_1,
                           K_prime,
                           L_prime,
                           r_prime_1,
                           w_prime_1,
                           Y_prime_1,
                           k_prime,
                           fw_prime_1,
                           linc_prime_1,
                           inc_prime_1], axis=1)

    # 2) Get tomorrow's policy
    # Tomorrow's capital: capital holding at beginning of period and how much
    # they save
    a_prime_1 = nn_predict(x_prime_1)
    a_prime_all_1 = tf.concat([a_prime_1, tf.zeros([m, 1])], axis=1)

    # 3) Tomorrow's consumption
    c_orig_prime_1 = inc_prime_1 - a_prime_all_1
    c_prime_1 = tf.maximum(c_orig_prime_1, tf.ones_like(c_orig_prime_1) * eps)

    # Shock 2 -----------------------------------------------------------------
    # 1) Get remaining parts of tomorrow's extended state
    # Exogenous shock
    z_prime_2 = 1 * tf.ones_like(z)

    # TFP and depreciation
    tfp_prime_2, depr_prime_2 = shocks(z_prime_2, eta, delta)

    # Return on capital, wage and aggregate production
    r_prime_2, w_prime_2, Y_prime_2 = firm(K_prime, tfp_prime_2, alpha, depr_prime_2)
    R_prime_2 = r_prime_2 * tf.ones([1, A])
    W_prime_2 = w_prime_2 * tf.ones([1, A])

    # D istribution of financial wealth, labor income, and total income
    fw_prime_2, linc_prime_2, inc_prime_2 = wealth(k_prime, R_prime_2, l_prime, W_prime_2)

    # Tomorrow's state: Concatenate the parts together
    x_prime_2 = tf.concat([tf.expand_dims(z_prime_2, -1),
                           tfp_prime_2,
                           depr_prime_2,
                           K_prime,
                           L_prime,
                           r_prime_2,
                           w_prime_2,
                           Y_prime_2,
                           k_prime,
                           fw_prime_2,
                           linc_prime_2,
                           inc_prime_2], axis=1)

    # 2) Get tomorrow's policy
    a_prime_2 = nn_predict(x_prime_2)
    a_prime_all_2 = tf.concat([a_prime_2, tf.zeros([m, 1])], axis=1)

    # 3) Tomorrow's consumption
    c_orig_prime_2 = inc_prime_2 - a_prime_all_2
    c_prime_2= tf.maximum(c_orig_prime_2, tf.ones_like(c_orig_prime_2) * eps)

    # Shock 3 -----------------------------------------------------------------
    # 1) Get remaining parts of tomorrow's extended state
    # Exogenous shock
    z_prime_3 = 2 * tf.ones_like(z)

    # TFP and depreciation
    tfp_prime_3, depr_prime_3 = shocks(z_prime_3, eta, delta)

    # Return on capital, wage and aggregate production
    r_prime_3, w_prime_3, Y_prime_3 = firm(K_prime, tfp_prime_3, alpha, depr_prime_3)
    R_prime_3 = r_prime_3 * tf.ones([1, A])
    W_prime_3 = w_prime_3 * tf.ones([1, A])

    # D istribution of financial wealth, labor income, and total income
    fw_prime_3, linc_prime_3, inc_prime_3 = wealth(k_prime, R_prime_3, l_prime, W_prime_3)

    # Tomorrow's state: Concatenate the parts together
    x_prime_3 = tf.concat([tf.expand_dims(z_prime_3, -1),
                           tfp_prime_3,
                           depr_prime_3,
                           K_prime,
                           L_prime,
                           r_prime_3,
                           w_prime_3,
                           Y_prime_3,
                           k_prime,
                           fw_prime_3,
                           linc_prime_3,
                           inc_prime_3], axis=1)

    # 2) Get tomorrow's policy
    # Tomorrow's capital: capital holding at beginning of period and how much
    # they save
    a_prime_3 = nn_predict(x_prime_3)
    a_prime_all_3 = tf.concat([a_prime_3, tf.zeros([m, 1])], axis=1)

    # 3) Tomorrow's consumption
    c_orig_prime_3 = inc_prime_3 - a_prime_all_3
    c_prime_3 = tf.maximum(c_orig_prime_3, tf.ones_like(c_orig_prime_3) * eps)

    # Shock 4 -----------------------------------------------------------------
    # 1) Get remaining parts of tomorrow's extended state
    # Exogenous shock
    z_prime_4 = 3 * tf.ones_like(z)

    # TFP and depreciation
    tfp_prime_4, depr_prime_4 = shocks(z_prime_4, eta, delta)

    # Return on capital, wage and aggregate production
    r_prime_4, w_prime_4, Y_prime_4 = firm(K_prime, tfp_prime_4, alpha, depr_prime_4)
    R_prime_4 = r_prime_4 * tf.ones([1, A])
    W_prime_4 = w_prime_4 * tf.ones([1, A])

    # D istribution of financial wealth, labor income, and total income
    fw_prime_4, linc_prime_4, inc_prime_4 = wealth(k_prime, R_prime_4, l_prime, W_prime_4)

    # Tomorrow's state: Concatenate the parts together
    x_prime_4 = tf.concat([tf.expand_dims(z_prime_4, -1),
                           tfp_prime_4,
                           depr_prime_4,
                           K_prime,
                           L_prime,
                           r_prime_4,
                           w_prime_4,
                           Y_prime_4,
                           k_prime,
                           fw_prime_4,
                           linc_prime_4,
                           inc_prime_4], axis=1)

    # 2) Get tomorrow's policy
    # Tomorrow's capital: capital holding at beginning of period and how much
    # they save
    a_prime_4 = nn_predict(x_prime_4)
    a_prime_all_4 = tf.concat([a_prime_4, tf.zeros([m, 1])], axis=1)

    # 3) Tomorrow's consumption
    c_orig_prime_4 = inc_prime_4 - a_prime_all_4
    c_prime_4 = tf.maximum(c_orig_prime_4, tf.ones_like(c_orig_prime_4) * eps)

    # Cost function ###########################################################
    # Prepare transitions to the next periods states. In this setting, there is
    # a 25% chance of ending up in any of the 4 states in Z. This has been
    # hardcoded and need to be changed to accomodate a different transition
    # matrix.
    pi_trans_to1 = p_transition * tf.ones((m, A-1))
    pi_trans_to2 = p_transition * tf.ones((m, A-1))
    pi_trans_to3 = p_transition * tf.ones((m, A-1))
    pi_trans_to4 = p_transition * tf.ones((m, A-1))

    # Euler equation
    opt_euler = - 1 + (
        (
            (
                beta * (
                    pi_trans_to1 * R_prime_1[:, 0:A-1] * c_prime_1[:, 1:A]**(-gamma) 
                    + pi_trans_to2 * R_prime_2[:, 0:A-1] * c_prime_2[:, 1:A]**(-gamma) 
                    + pi_trans_to3 * R_prime_3[:, 0:A-1] * c_prime_3[:, 1:A]**(-gamma) 
                    + pi_trans_to4 * R_prime_4[:, 0:A-1] * c_prime_4[:, 1:A]**(-gamma)
                )
            ) ** (-1. / gamma)
        ) / c[:, 0:A-1]
    )

    # Punishment for negative consumption (c)
    orig_cons = tf.concat([c_orig, c_orig_prime_1, c_orig_prime_2, c_orig_prime_3, c_orig_prime_4], axis=1)
    opt_punish_cons = (1./eps) * tf.maximum(-1 * orig_cons, tf.zeros_like(orig_cons))

    # Punishment for negative aggregate capital (K)
    opt_punish_ktot_prime = (1./eps) * tf.maximum(-K_prime_orig, tf.zeros_like(K_prime_orig))

    # Concatenate the 3 equilibrium functions
    combined_opt = [opt_euler, opt_punish_cons, opt_punish_ktot_prime]
    opt_predict = tf.concat(combined_opt, axis=1)

    # Define the "correct" outputs. For all equilibrium functions, the correct
    # outputs is zero.
    opt_correct = tf.zeros_like(opt_predict)

    # Define the cost function
    cost = tf.losses.mean_squared_error(opt_correct, opt_predict)

    # Optimizer and gradient descent ##########################################
    # Adam optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    # Clip the gradients to limit the extent of exploding gradients
    gvs = optimizer.compute_gradients(cost)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]

    # Define a training step
    train_step = optimizer.apply_gradients(capped_gvs)

    # ####################################################################### #
    #                                Training                                 #
    # ####################################################################### #
    def simulate_episodes(sess, x_start, episode_length, print_flag=True):
        """Simulate an episode for a given starting point using the current
        neural network state.

        Args:
            sess: current tensorflow session,
            x_start: starting state to simulate forward from,
            episode_length: number of steps to simulate forward,
            print_flag: boolean that determines whether to print simulation stats.

        Returns:
            X_episodes: tensor of states [z, k] to train on (training set).
        """
        time_start = datetime.now()
        if print_flag:
            print('Start simulating {} periods.'.format(episode_length))
        dim_state = np.shape(x_start)[1]

        X_episodes = np.zeros([episode_length, dim_state])
        X_episodes[0, :] = x_start
        X_old = x_start

        # Generate a sequence of random shocks
        rand_num = np.random.rand(episode_length, 1)

        for t in range(1, episode_length):
            z = int(X_old[0, 0])  # Current period's shock

            # Determine which state we will be in in the next period based on
            # the shock and generate the corresponding state (x_prime)
            if rand_num[t - 1] <= pi_np[z, 0]:
                X_new = sess.run(x_prime_1, feed_dict={X: X_old})
            elif rand_num[t - 1] <= pi_np[z, 0] + pi_np[z, 1]:
                X_new = sess.run(x_prime_2, feed_dict={X: X_old})
            elif rand_num[t - 1] <= pi_np[z, 0] + pi_np[z, 1] + pi_np[z, 2]:
                X_new = sess.run(x_prime_3, feed_dict={X: X_old})
            else:
                X_new = sess.run(x_prime_4, feed_dict={X: X_old})
            
            # Append it to the dataset
            X_episodes[t, :] = X_new
            X_old = X_new

        time_end = datetime.now()
        time_diff = time_end - time_start
        if print_flag:
            print('Finished simulation. Time for simulation: {}.'.format(time_diff))

        return X_episodes

    # Analytical solution #####################################################
    # Get the analytical solution
    beta_vec = beta_np * (1 - beta_np ** (A - 1 - np.arange(A-1))) / (1 - beta_np ** (A - np.arange(A-1)))
    beta_vec = tf.constant(np.expand_dims(beta_vec, 0), dtype=tf.float32)
    a_analytic = inc[:, : -1] * beta_vec

    # Training the deep equilibrium net #######################################
    # Helper variables for plotting
    all_ages = np.arange(1, A+1)
    ages = np.arange(1, A)

    # Initialize tensorflow session
    sess = tf.Session()

    # Generate a random starting point
    if args.load_episode == 0:
        X_data_train = np.random.rand(1, num_input_nodes)
        X_data_train[:, 0] = (X_data_train[:, 0] > 0.5)
        X_data_train[:, 1:] = X_data_train[:, 1:] + 0.1
        assert np.min(np.sum(X_data_train[:, 1:], axis=1, keepdims=True) > 0) == True, 'Starting point has negative aggregate capital (K)!'
        print('Calculated a valid starting point')
    else:
        data_path = './output/startpoints/data_{}.npy'.format(args.load_episode)
        X_data_train = np.load(data_path)
        print('Loaded initial data from ' + data_path)

    train_seed = 0

    cost_store, mov_ave_cost_store = [], []

    time_start = datetime.now()
    print('start time: {}'.format(time_start))

    # Initialize the random variables (neural network weights)
    init = tf.global_variables_initializer()

    # Initialize saver to save and load previous sessions
    saver = tf.train.Saver()

    # Run the initializer
    sess.run(init)

    if args.load_episode != 0:
        saver.restore(sess, './output/models/sess_{}.ckpt'.format(args.load_episode))
                
    for episode in range(args.load_episode, num_episodes):
        # Simulate data: every episode uses a new training dataset generated on
        # the current iteration's neural network parameters.
        X_episodes = simulate_episodes(sess, X_data_train, len_episodes, print_flag=(episode==0))
        X_data_train = X_episodes[-1, :].reshape([1, -1])
        k_dist_mean = np.mean(X_episodes[:, 8 : 8 + A], axis=0)
        k_dist_min = np.min(X_episodes[:, 8 : 8 + A], axis=0)
        k_dist_max = np.max(X_episodes[:, 8 : 8 + A], axis=0)
        
        ee_error = np.zeros((1, num_agents-1))
        max_ee = np.zeros((1, num_agents-1))

        for epoch in range(epochs_per_episode):
            # Every epoch is one full pass through the dataset. We train
            # multiple passes on one training set before we resimulate a
            # new dataset.
            train_seed += 1
            minibatch_cost = 0

            # Mini-batch the simulated data
            minibatches = random_mini_batches(X_episodes, minibatch_size, train_seed)

            for minibatch_X in minibatches:
                # Run optimization; i.e., determine the cost of each mini-batch.
                minibatch_cost += sess.run(cost, feed_dict={X: minibatch_X}) / num_minibatches
                if epoch == 0:
                    # For the first epoch, save the mean and max euler errors for plotting
                    # This way, the errors are calculated out-of-sample.
                    opt_euler_ = np.abs(sess.run(opt_euler, feed_dict={X: minibatch_X}))
                    ee_error += np.mean(opt_euler_, axis=0) / num_minibatches
                    mb_max_ee = np.max(opt_euler_, axis=0, keepdims=True)
                    max_ee = np.maximum(max_ee, mb_max_ee)

            if epoch == 0:
                # Record the cost and moving average of the cost at the beginning of each
                # episode to track learning progress.
                cost_store.append(minibatch_cost)
                mov_ave_cost_store.append(np.mean(cost_store[-100:]))

            for minibatch_X in minibatches:
                # Take a mini-batch gradient descent training step. That is, update the
                # weights for one mini-batch.
                sess.run(train_step, feed_dict={X: minibatch_X})

        if episode % args.plot_interval == 0:
            # Plot
            # Plot the loss function
            plt.figure(figsize=std_figsize)
            ax = plt.subplot(1,1,1)
            ax.plot(np.log10(cost_store), 'k-', label='cost')
            ax.plot(np.log10(mov_ave_cost_store), 'r--', label='moving average')
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Cost [log10]')
            ax.legend(loc='upper right')
            plt.savefig('./output/plots/loss_ep_%d.pdf' % episode, bbox_inches='tight')
            plt.close()

            # Plot the relative errors in the Euler equation
            plt.figure(figsize=std_figsize)
            ax = plt.subplot(1,1,1)
            ax.plot(ages, np.log10(ee_error).ravel(), 'k-', label='mean')
            ax.plot(ages, np.log10(max_ee).ravel(), 'k--', label='max')
            ax.set_xlabel('Age')
            ax.set_ylabel('Rel EE [log10]')
            ax.legend()
            plt.savefig('./output/plots/relee_ep_%d.pdf' % episode, bbox_inches='tight')
            plt.close()

            # Plot the capital distribution
            plt.figure(figsize=std_figsize)
            ax = plt.subplot(1,1,1)
            ax.plot(all_ages, k_dist_mean, 'k-', label='mean')
            ax.plot(all_ages, k_dist_min, 'k-.', label='min')
            ax.plot(all_ages, k_dist_max, 'k--', label='max')
            ax.set_xlabel('Age')
            ax.set_ylabel('Capital (k)')
            ax.legend()
            ax.set_xticks(all_ages)
            plt.savefig('./output/plots/distk_ep_%d.pdf' % episode, bbox_inches='tight')
            plt.close()
            
            # =======================================================================================
            # Sample 50 states and compare the neural network's prediction to the analytical solution
            pick = np.random.randint(len_episodes, size=50)
            random_states = X_episodes[pick, :]

            # Sort the states by the exogenous shock
            random_states_1 = random_states[random_states[:, 0] == 0]
            random_states_2 = random_states[random_states[:, 0] == 1]
            random_states_3 = random_states[random_states[:, 0] == 2]
            random_states_4 = random_states[random_states[:, 0] == 3]

            # Get corresponding capital distribution for plots
            random_k_1 = random_states_1[:, 8 : 8 + A]
            random_k_2 = random_states_2[:, 8 : 8 + A]
            random_k_3 = random_states_3[:, 8 : 8 + A]
            random_k_4 = random_states_4[:, 8 : 8 + A]

            # Generate a prediction using the neural network
            nn_pred_1 = sess.run(a, feed_dict={X: random_states_1})
            nn_pred_2 = sess.run(a, feed_dict={X: random_states_2})
            nn_pred_3 = sess.run(a, feed_dict={X: random_states_3})
            nn_pred_4 = sess.run(a, feed_dict={X: random_states_4})

            # Calculate the analytical solution
            true_pol_1 = sess.run(a_analytic, feed_dict={X: random_states_1})
            true_pol_2 = sess.run(a_analytic, feed_dict={X: random_states_2})
            true_pol_3 = sess.run(a_analytic, feed_dict={X: random_states_3})
            true_pol_4 = sess.run(a_analytic, feed_dict={X: random_states_4})

            # Plot both
            for i in range(A - 1):
                plt.figure(figsize=std_figsize)
                ax = plt.subplot(1,1,1)
                # Plot the true solution with a circle
                ax.plot(random_k_1[:, i], true_pol_1[:, i], 'ro', mfc='none', alpha=0.5, markersize=6, label='analytic')
                ax.plot(random_k_2[:, i], true_pol_2[:, i], 'bo', mfc='none', alpha=0.5, markersize=6)
                ax.plot(random_k_3[:, i], true_pol_3[:, i], 'go', mfc='none', alpha=0.5, markersize=6)
                ax.plot(random_k_4[:, i], true_pol_4[:, i], 'yo', mfc='none', alpha=0.5, markersize=6)
                # Plot the prediction of the neural net
                ax.plot(random_k_1[:, i], nn_pred_1[:, i], 'r*', markersize=2, label='DEQN')
                ax.plot(random_k_2[:, i], nn_pred_2[:, i], 'b*', markersize=2)
                ax.plot(random_k_3[:, i], nn_pred_3[:, i], 'g*', markersize=2)
                ax.plot(random_k_4[:, i], nn_pred_4[:, i], 'y*', markersize=2)
                ax.set_title('Agent {}'.format(i+1))
                ax.set_xlabel(r'$k_t$')
                ax.set_ylabel(r'$a_t$')
                ax.legend() 
                plt.savefig('./output/plots/policy_agent_%d_ep_%d.pdf' % (i+1, episode), bbox_inches='tight')
                plt.close()
            
            
        # Print cost and time log
        print('Episode {}: \t log10(Cost): {:.4f}; \t runtime: {}'\
            .format(episode, np.log10(cost_store[-1]), datetime.now()- time_start))

        if episode % args.save_interval == 0:
            # Save the tensorflow session
            saver.save(sess, './output/models/sess_{}.ckpt'.format(episode))
            # Save the starting point
            np.save('./output/startpoints/data_{}.npy'.format(episode), X_data_train)

if __name__ == '__main__':
    global args
    args = parser.parse_args()
    print(args)

    # Make output directory to save network weights and starting point
    if not os.path.exists('./output'):
        os.mkdir('./output')

    if not os.path.exists('./output/models'):
        os.mkdir('./output/models')

    if not os.path.exists('./output/startpoints'):
        os.mkdir('./output/startpoints')

    if not os.path.exists('./output/plots'):
        os.mkdir('./output/plots')

    print('Plots will be saved into ./output/plots/.')

    # Train deep equilibrium net
    main(args)

