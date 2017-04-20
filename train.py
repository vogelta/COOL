from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cnn_classifier import Parameters, CNNClassifier, load_cnn
from transfer import preprocess, retrain_layers


def train_new_cnn(name):
  """
  Train a new CNNClassifier model according to the parameters in
  './Parameters/<name>'.
  """
  g = tf.Graph()
  with g.as_default():
    params = Parameters(name)
    CNN = CNNClassifier(params)
    CNN.train_model()
  
def train_transfer_cnn(name):
  """
  Train a CNNClassifier model by transfer-learning from a pretrained
  model, according to the parameters in './Parameters/<name>'.
  """
  retrain_layers(name)

def show_dead_units(layer, name, run=None, threshold=0):
  """
  Processes data through a pretrained model with <name> (using weights
  from <run>), up through layer number (zero-indexed) <layer> and 
  reports how many units were never above <threshold> on the training 
  set: with value 0 after a ReLU activation, these are 'dead' units.
  
  Returns:
  - an array with how many times each member of the layer was
    above <threshold> on the training set.
  - the preprocessed dataset.
  """
  new_wrap, new_dim = preprocess(layer+1, name, run)
  new_data = new_wrap()
  images_on = np.sum(np.greater(new_data.train.images, threshold), 0)
  images_on = images_on.reshape([-1])
  n_dead = np.sum(np.equal(images_on, 0))
  print('{:d} Dead Units'.format(n_dead))
  plt.plot(np.sort(images_on))
  plt.title(name)
  plt.xlabel('Layer {:d} Units (Sorted)'.format(layer))
  plt.ylabel('Training Examples Above {:.2f}'.format(threshold))
  plt.show()
  return images_on, new_data

def cool_to_softmax(name, run=None):
  """
  With a trained model with COOL or MinCOOL output layer, takes one of
  the 'Overcomplete' units from each class and evaluates the test-set
  performance of a softmax layer using those units.
  
  Assumes the model named actually did use a COOL final layer. Also 
  assumes the tf.Variables for 'weights' and 'biases' used those names 
  in the final fully-connected layer.
  """
  g = tf.Graph()
  with g.as_default(), tf.Session() as session:
    CNN = load_cnn(session, name, run)
    n_layers = len(CNN.params.layers)
    doo = CNN.params.layers[n_layers-1].fn_params['doo']
    if isinstance(doo, int):
      doo = [doo]*CNN.params.n_classes
    with tf.variable_scope(name+'/Layer_{:d}'.format(n_layers-2), reuse=True):
      W = tf.get_variable('weights')
      B = tf.get_variable('biases')
    new_W = []
    new_B = []
    start = 0
    for d in doo:
      idx = np.random.randint(d)
      new_W.append(W[:,start+idx])
      new_B.append(B[start+idx])
      start += d
    new_W = tf.pack(new_W, 1)
    new_B = tf.pack(new_B, 0)
    
    x = tf.matmul(CNN.layer_outputs[-3], new_W) + new_B
    out = tf.nn.softmax(x)
    loss, accuracy = CNN.add_loss(out)
    
    data = CNN.params.data_wrap()
    feed = {CNN.input_placeholder: data.test.images, 
            CNN.labels_placeholder: data.test.labels}
    loss_, acc_ = session.run([loss, accuracy], feed_dict=feed)
  print('Loss {:.5f}, Accuracy {:.4f}'.format(loss_, acc_))
    
