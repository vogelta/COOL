from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from data_wrappers import mnist_wrapper
from utils import Layer, cool_layer


data_wrap = mnist_wrapper
image_dim = [28,28,1]
n_classes = 10

batch_size = 50
n_batches = 40000
check_every = 1000

optimizer = tf.train.AdamOptimizer
learning_rate = 4e-4
decay_function = tf.train.piecewise_constant
decay_params = [[20000], [learning_rate, 2e-4]]

w_init = tf.truncated_normal_initializer(stddev=0.1)
b_init = tf.constant_initializer(0.1)
doo = 5

layers_to_track_dead_units = [0,2,5]
layers = [Layer(tf.contrib.layers.convolution2d,
                {'num_outputs': 32, 
                 'kernel_size': 5, 
                 'activation_fn': tf.nn.relu,
                 'weights_initializer': w_init,
                 'biases_initializer': b_init,
                 'trainable': True}),
          Layer(tf.contrib.layers.max_pool2d, {'kernel_size': 2}),
          Layer(tf.contrib.layers.convolution2d,
                {'num_outputs': 64, 
                 'kernel_size': 5, 
                 'activation_fn': tf.nn.relu,
                 'weights_initializer': w_init,
                 'biases_initializer': b_init,
                 'trainable': True}),
          Layer(tf.contrib.layers.max_pool2d, {'kernel_size': 2}),
          Layer(tf.contrib.layers.flatten),
          Layer(tf.contrib.layers.fully_connected,
                {'num_outputs': 1024,
                 'activation_fn': tf.nn.relu,
                 'weights_initializer': w_init,
                 'biases_initializer': b_init,
                 'trainable': True}),
          Layer(tf.contrib.layers.fully_connected,
                {'num_outputs': n_classes*doo,
                 'activation_fn': tf.nn.softmax,
                 'weights_initializer': w_init,
                 'biases_initializer': b_init,
                 'trainable': True}),
          Layer(cool_layer, {'doo': doo, 'mode': 'min'}, ['is_training'])]
