"""Utils for building components of a network."""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def weight_variable(w_dim, name, trainable=True, dtype=tf.float32):
    """Create weights.

    Args:
        w_dim (list): [input dimension, output dimension]
    """
    t_stnd = (tf.sqrt(tf.cast(w_dim[0], dtype)) * 1000)
    return tf.Variable(tf.random_normal(w_dim) / t_stnd, name=name, trainable=trainable)


def bias_variable(b_dim, name, trainable=True, dtype=tf.float32):
    """Create biases."""
    t_stnd = tf.cast(b_dim[0], dtype) * 1000
    return tf.Variable(tf.random_normal(b_dim) / t_stnd, name=name, trainable=trainable)


def linear_combination(x, w, b):
    """Create weighted linear combination of X.

    Args:
        kwargs takes mu, sd, and eps.
        #mu and sd are optional but must be passedtogether
        sd and eps are optional
        eps defaults to 0.00001.
    """
    return tf.add(tf.matmul(x, w), b, name='pre_activation')


class Neural_Net:
    def __init__(self, num_nodes,
                 activation_list,
                 weight_init_method=None,
                 bias_init_method=None,
                 trainable=True,
                 dtype=tf.float32):
        assert len(activation_list) == len(num_nodes) - 1, 'length of activation_list doesnt match the number of layers'

        self.num_layers = len(num_nodes)
        print('{} hidden layers in NN'.format(self.num_layers-2))
        for i in range(1, self.num_layers):
            if i == self.num_layers-1:
                print('Output layer: nodes: '+str(num_nodes[i])+', activation: '+str(activation_list[i-1]))
            else:
                print('Hidden layer '+str(i)+': nodes: '+str(num_nodes[i])+', activation: '+str(activation_list[i-1]))
        self.num_nodes = num_nodes
        self.n_input, self.n_output = num_nodes[0], num_nodes[-1]
        self.activation_list = activation_list
        self.param_dict = {}
        with tf.name_scope('network_parameters'):
            if weight_init_method is None and bias_init_method is None:
                for i in range(self.num_layers-1):
                    nm_suffix = str(i)
                    w_dim = [num_nodes[i], num_nodes[i+1]]
                    b_dim = [num_nodes[i+1]]
                    with tf.name_scope('weights'):
                        self.param_dict['w' + nm_suffix] = weight_variable(w_dim, 'w' + nm_suffix, trainable, dtype)
                        tf.summary.histogram('w'+nm_suffix, self.param_dict['w' + nm_suffix])
                    with tf.name_scope('biases'):
                        self.param_dict['b' + nm_suffix] = bias_variable(b_dim, 'b' + nm_suffix, trainable, dtype)
                        tf.summary.histogram('b'+nm_suffix, self.param_dict['b' + nm_suffix])
            else:
                raise NotImplementedError

    def predict(self, net_input, dtype=tf.float32):
        x = tf.cast(net_input, dtype)
        tf.summary.histogram('net_input', x)
        for i in range(self.num_layers-1):
            x = linear_combination(x, self.param_dict['w' + str(i)], self.param_dict['b' + str(i)])
            act = self.activation_list[i]
            if act is not None:
                x = act(x)
        return x
