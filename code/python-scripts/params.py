"""Model and network calibration (hyper)parameters."""
import tensorflow as tf
import numpy as np

# ####################################################################### #
#                            Model calbration                             #
# ####################################################################### #
# Number of agents and exogenous shocks
num_agents = A = 6  # Do not change, this is hardcoded

# Exogenous shock values
# Capital depreciation (dependent on shock)
delta = tf.constant([[0.5], [0.5], [0.9], [0.9]],  dtype=tf.float32)
# TFP shock (dependent on shock)
eta = tf.constant([[0.95], [1.05], [0.95], [1.05]], dtype=tf.float32)

# Transition matrix
# In this example we hardcoded the transition matrix. Changes cannot be made without also changing
# the corresponding code below.
p_transition = 0.25  # All transition probabilities are 0.25
pi_np = p_transition * np.ones((4, 4))  # Transition probabilities
pi = tf.constant(pi_np, dtype=tf.float32)  # Transition probabilities

# Labor endowment
labor_endow_np = np.zeros((1, A))
labor_endow_np[:, 0] = 1.0  # Agents only work in their first period
labor_endow = tf.constant(labor_endow_np, dtype=tf.float32)

# Production and household parameters
alpha = tf.constant(0.3)  # Capital share in production
beta_np = 0.7
beta = tf.constant(beta_np, dtype=tf.float32)  # Discount factor (patience)
gamma = tf.constant(1.0, dtype=tf.float32)  # CRRA coefficient

# ####################################################################### #
#                             Neural network                              #
# ####################################################################### #
# Neural network training hyperparameters ---------------------------------
num_episodes = 5000
len_episodes = 10240
epochs_per_episode = 20
minibatch_size = 512
num_minibatches = int(len_episodes / minibatch_size)
lr = 0.00001

# Neural network architecture parameters ----------------------------------
num_input_nodes = 8 + 4 * A   # Dimension of state space (1 shock and A agents)
num_hidden_nodes = [100, 50]  # Dimension of hidden layers
num_output_nodes = (A - 1)    # Output dimension (capital holding for each
                              # agent. Agent 1 is born without capital (k^1=0))
