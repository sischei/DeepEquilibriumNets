"""Helper functions for the jupyter notebooks."""

from datetime import datetime
import math
import numpy as np
import tensorflow as tf


def initialize_nn_weight(dim):
    """Initialize neural network weight or bias variable.

    Args:
        dim <list>: dimensions of weight matrix.
                    - if len(dim) == 1: initializes bias,
                    - if len(dim) == 2: initializes weight.

    Returns:
        Tensor variable of normally initialized data.

    """
    t_stnd = tf.sqrt(tf.cast(dim[0], tf.float32)) * 10
    return tf.Variable(tf.random_normal(tf.cast(dim, tf.int32)) / t_stnd, trainable=True)


def random_mini_batches(X, minibatch_size=64, seed=0):
    """Generate random minibatches from X.

    Args:
        X <array>: Input data to be mini-batched.
        minibatch_size <int>: mini-batch size.
        seed <int>: seed.

    Returns:
        List of mini-batches generated from X.

    """
    np.random.seed(seed)
    m = X.shape[0]
    mini_batches = []

    # Step 1: Shuffle X
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]

    # Step 2: Partition shuffled_X. Minus the end case.
    # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = int(math.floor(m / minibatch_size))

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[(k * minibatch_size):((k+1) * minibatch_size), :]
        mini_batch = (mini_batch_X)
        mini_batches.append(mini_batch)

    return mini_batches
