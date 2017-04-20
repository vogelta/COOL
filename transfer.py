from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
import time

from cnn_classifier import Parameters, CNNClassifier, load_cnn

def preprocess(n_layers, modelname, run=None, data=None, block_size=10000):
  """
  Taking a CNNClassifier model, modify a dataset by replacing each
  image with the image's representation after being processed up to 
  the first n_layers of the model.
  
  Inputs:
  - n_layers: int, how many layers to feedforward through
  - modelname: name of the model
  - run: optional, which of the model's training runs weights to use:
      can be int, number of the run, or list/tuple of that number and 
      'BEST' or 'LAST'
  - data: the dataset to process in the form specified by 
      data_wrappers.py, images must be the same shape as those that 
      'model' was trained on. If None, uses the data_wrap parameter of 
      'model'.
  - block_size: optional int, maximum number of train-images to process
      at once
  Returns:
  - new_wrap: function wrapping the modified dataset
  - new_dim: list, the shape of the modified images
  """
  start_time = time.time()
  g = tf.Graph()
  with g.as_default(), tf.Session() as session:
    model = load_cnn(session, modelname, run)
    if data is None:
      data = model.params.data_wrap()
    start = 0
    train_size = data.train.images.shape[0]
    train_images = []
    while start < train_size:
      end = min(start+block_size, train_size)
      batch = data.train._images[start:end,]
      print('Processing training images {:d} - {:d}'.format(start, end))
      feed = {model.input_placeholder: batch}
      newbatch = session.run(model.layer_outputs[n_layers-1], feed_dict=feed)
      train_images.append(newbatch)
      start = end
    data.train._images = np.concatenate(train_images, axis=0)
  
    print('Processing validation images')
    feed = {model.input_placeholder: data.validation._images}
    data.validation._images = session.run(model.layer_outputs[n_layers-1],
                                          feed_dict=feed)
    print('Processing test images')
    feed = {model.input_placeholder: data.test._images}
    data.test._images = session.run(model.layer_outputs[n_layers-1], 
                                    feed_dict=feed)
  
  new_dim = list(data.train._images.shape[1:])
  def new_wrap():
    return data
  print('Preprocessing time: {:.2f}'.format(time.time()-start_time))
  return new_wrap, new_dim
  
def retrain_layers(finalname):
  """
  Trains a new CNNClassifer object according to the parameters in
  ./Parameters/<finalname>. 
  
  The up to layer <up_to_layer> is taken from a previously-trained
  CNNClassifier named <original_model>. The weights are treated as
  frozen so the dataset is preprocessed through these layers for 
  efficiency. 
  
  The Parameter class automatically appended the original model's 
  layers to its layer attribute, so those have to be removed again 
  before the new training process.
  
  The <layers> of the new model are then trained on the preprocessed 
  data. After training, the new model's saved weights are updated in 
  such a way that when they are loaded with cnn_classifier.load_cnn(),
  they apply to the original non-preprocessed dataset.
  """
  finalparams = Parameters(finalname)
  
  g = tf.Graph()
  with g.as_default(), tf.Session() as session:
    new_wrap, new_dim = preprocess(finalparams.up_to_layer+1,
                                   finalparams.original_model, 
                                   finalparams.original_run)
    
    finalparams.data_wrap = new_wrap
    finalparams.image_dim = new_dim
    finalparams.layers = finalparams.layers[(finalparams.up_to_layer+1):]
    
    new_model = CNNClassifier(finalparams)
    new_model.train_model()
    with tf.variable_scope(finalname):
      new_model.build_model()
    
    new_vars = g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=finalname)
    new_model_loader = tf.train.Saver(new_vars)
    
    original = load_cnn(session, finalparams.original_model, 
                        finalparams.original_run)
    old_vars = g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                scope=finalparams.original_model)
    
    var_dict = {}
    for var in new_vars:
      name = var.op.name
      n = name.find('/Layer_')
      if n != -1:
        n = int(name[n+7])
        name = name.replace('/Layer_{:d}'.format(n), 
                            '/Layer_{:d}'.format(n+finalparams.up_to_layer+1))
      var_dict[name] = var
    for var in old_vars:
      name = var.op.name
      n = name.find('/Layer_')
      if n != -1:
        n = int(name[n+7])
        if n <= finalparams.up_to_layer:
          name = name.replace(finalparams.original_model, finalparams.name)
          var_dict[name] = var
    
    modelsaver = tf.train.Saver(var_dict)
    saves_to_update = [finalparams.make_weights_path('LAST')]
    if finalparams.check_every <= finalparams.n_batches:
      saves_to_update.append(finalparams.weights_path)
    for save_file in saves_to_update:
      new_model_loader.restore(session, save_file)
      modelsaver.save(session, save_file)

