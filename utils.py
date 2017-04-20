from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six.moves as sm
import tensorflow as tf
import os


class Layer(object):
  """
  Class for wrapping neural network layers such that we can define the
  hyperparameters in __init__() while actually creating the graph 
  elements in __call__()
  """
  def __init__(self, layer_fn, fn_params={}, placeholder_args=[]):
    """
    Inputs:
    - layer_fn: function calculating the layer outputs. Always takes 
        arguments inputs and scope and any others in given fn_params.
    - placeholder_args: a list of the names of placeholders the layer
        will take as arguments, eg. ['keep_prob'] for dropout layers
    - fn_params: dict of any other arguments to feed to layer_fn
    """
    self.layer_fn = layer_fn
    self.fn_params = fn_params
    self.placeholder_args = placeholder_args
    
  def __call__(self, x, extra_holders, scope):
    """
    Inputs:
    - x: inputs to the layer
    - extra_holders: a dict mapping argument-names to the corresponding
        tf.placeholders (not including the input_placeholder)
    - scope: string for tf.name_scope
    """
    for holder in self.placeholder_args:
      if isinstance(holder, str):
        self.fn_params[holder] = extra_holders[holder]
      else:
        if isinstance(self.fn_params[holder[0]], dict):
          self.fn_params[holder[0]][holder[1]] = extra_holders[holder[1]]
        else:
          self.fn_params[holder[0]] = {holder[1]: extra_holders[holder[1]]}
    return self.layer_fn(inputs=x, scope=scope, **self.fn_params)
  

def cool_layer(inputs, doo, is_training=True, mode='geometric',
               eps=1e-15, scope=None):
  """
  Computes the Competitive Overcomplete Output Layer.
  
  Geometric mean calculation has been arranged for numerical stability,
  including use of optional input eps. 
    
  Inputs:
  - inputs: a 2-D tf.Tensor, assumed to be the output from a Softmax
      layer. AssertionError caused if width is not compatible with doo
  - doo: int or list, the degrees-of-overcompleteness. In the case of
      an int, every class uses that same value.
  - is_training: bool, or tf.Tensor with type tf.bool, optional input,
      when False outputs are scaled such that they sum to 1 (in order
      to minimize loss), otherwise they are left unscaled.
  - mode: 'geometric' or 'min', which function is used to combine units
      in each aggregate - taking the geometric mean or minimum.
  - scope: string for tf.name_scope
  
  The is_training scaling is OFF by default. To use it, a placeholder 
  with value 'True' should be fed during training and 'False' in 
  evaluation.
  """
  with tf.name_scope(scope, 'COOL', [inputs]):
    assert mode in ['geometric', 'min']
    is_ = inputs.get_shape().as_list()[1]
    if isinstance(doo, list):
      assert is_ == sum(doo)
    else:
      assert isinstance(doo, int)
      assert is_ % doo == 0
      doo = [doo]*int(is_/doo)
    
    outputs = []
    begin = [0, 0]
    for degree in doo:
      section = tf.slice(inputs, begin, [-1,degree]) + eps
      if mode is 'geometric':
        result = tf.reduce_sum(tf.log(float(degree)*section), 1, keep_dims=True)
        result = tf.exp(result/float(degree))
      else:
        result = float(degree)*tf.reduce_min(section, 1, keep_dims=True)
      outputs.append(result)
      begin[1] += degree
    outputs = tf.concat(1, outputs)
    if isinstance(is_training, bool):
      is_training = tf.convert_to_tensor(is_training)
    def f1():
      return outputs
    def f2():
      output_sum = tf.reduce_sum(outputs, axis=1, keep_dims=True)
      return outputs/tf.maximum(output_sum, 1e-15)
    return tf.cond(is_training, f1, f2)

def one_hot(x, n_classes):
  """
  Turns list-of-labels x into one-hot form having a total of n_classes.
  """
  out = np.zeros((len(x), n_classes))
  out[sm.range(len(x)),x] = 1.
  return out

def sequential_name(folder, basename):
  """
  Given a proposed name for a file (string 'basename') to be saved in a
  folder (identified by its path in string 'folder'), produces a new
  name to use that avoids overwriting other files - as long as their 
  names were made with this function, too.
  """
  if not os.access(folder, os.F_OK):
    return '{:s}/{:s}'.format(folder, basename)
  else:
    existing_files = os.listdir(folder)
    matches = sum([basename in x for x in existing_files])
    if matches == 0:
      return '{:s}/{:s}'.format(folder, basename)
    else:
      return '{:s}/{:s} ({:d})'.format(folder, basename, matches)

