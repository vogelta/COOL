from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import six.moves as sm
import tensorflow as tf
import time
from sys import path, modules

from utils import sequential_name

path.insert(1, './Parameters')


class Parameters(object):
  """
  A Parameters-class object takes as input a string naming a .py file
  in the ./Parameters folder from which attributes will be retrieved.
  This string will also name the model for saving weights, training 
  logs and any fooling-images created.
  
  Attributes in <required_vars> must be included, while <optional_vars>
  are optional. There are also other attributes that will be retrieved
  based on whether others are present, e.g. decay_params if a 
  decay_function is given.
  
  required_vars
  - data_wrap: a function that returns the dataset to use when called,
        expected to be in the format given in data_wrappers.py
  - n_classes: int, number of output classes
  - image_dim: list-of-ints, size of the input images as [H,W,C]
  - batch_size: int, size of training batches
  - n_batches: int, maximum number of batches to train for
  - check_every: int, in batches, how often to report validation error,
        update logs, and check for early stopping (if used)
  - optimizer: tf.train.Optimizer class to use for training
  - learning_rate: float, learning rate for optimizer. Used as the 
        initial rate with LR decay (except piecewise_constant)
  - layers: list of layers of the network, in the format of Layer class
        of utils.py
  
  optional_vars
  - early_stopping: int, maximum number of batches since the best 
        validation result before training is stopped
  - clip_min_and_max: list of floats, [min, max] to use for gradient 
        clipping
  - decay_function: function to use for learning rate decay
  - dropout_prob: float, dropout probability to use in training
  - layers_to_track_dead_units: list-of-ints giving (zero-indexed) 
        indices of layers where the number of inactive output units 
        will be logged (intended for use with ReLU activation function)
  - original_model: string, for transfer learning, the name of the 
        pretrained model to use weights from
  - train_loss_check_every: int, in batches, how often to log training
        loss. Defaults to <check_every>.
  
  dependent on decay_function:
  - decay_params: list of parameter values to feed to decay_function,
        aside from initial rate and global_step.
  
  dependent on original_model:
  - up_to_layer: int, the final layer from the original model to take.
        All layers from before that one are taken too: layers are zero-
        indexed, so up_to_layer=3 means the first 4 layers are used. 
  - original_run: int, or tuple of (int, 'BEST' or 'LAST') giving which
        training run of the original model to use
  - also, if original_model was given, the layers parameter is changed
    by appending to the start of the list, the layers from the original
    model. Thus self.layers represents all the layers used in taking 
    the raw image to the final classification.
  """
  required_vars = [
              'data_wrap',
              'n_classes',
              'image_dim',
              'batch_size',
              'n_batches',
              'check_every', 
              'optimizer',
              'learning_rate',
              'layers']
  optional_vars = [
              'early_stopping',
              'clip_min_and_max',
              'decay_function',
              'dropout_prob',
              'layers_to_track_dead_units',
              'original_model',
              'train_loss_check_every']
              
  def __init__(self, name, run=None):
    """
    The optional input <run> can be used to identify a specific
    training run of the CNNClassifier model defined by <name>.
    
    Both the best-validation-loss ('BEST') and last-found ('LAST') 
    weights are saved during training. By default, the best weights of
    the most-recent run (that with the highest number) are accessed.
    
    <run> can be given as an integer of the run number or as a tuple 
    of (int, 'BEST' or 'LAST'). In the integer case, 'BEST' will be
    assumed (warning: if the training run had lower n_batches than 
    check_every parameter, the 'BEST' save will not have been created).
    
    Training a new model in CNNClassifier assumes run was given as None 
    as it will save under a new run with its number incremented by one.
    """
    if run is None:
      self.run_was_set = False
      if os.access('./Weights/{:s}'.format(name), os.F_OK):
        entries = os.listdir('./Weights/{:s}'.format(name))
        entries = [int(x[4:]) for x in entries if x.startswith('Run_')] + [0]
        self.run = max(entries)
      else:
        self.run = 0
      version = 'BEST'
    elif isinstance(run,int):
      self.run_was_set = True
      self.run = run
      version = 'BEST'
    else:
      self.run_was_set = True
      assert isinstance(run[0], int)
      self.run = run[0]
      version = run[1].upper()
      assert version in ['BEST', 'LAST']
    
    self.name = name
    self.weights_path = self.make_weights_path(version)
    self.logs_path = self.make_logs_path()
    
    if self.run_was_set:
      folder, _, filename = self.weights_path.rpartition('/')
      weights = os.listdir(folder)
      if filename+'.data-00000-of-00001' not in weights:
        raise ValueError('{:s} no weights saved for this run')
    
    try:
      param_file = modules[name]
      sm.reload_module(param_file)
    except KeyError:
      param_file = __import__(name)
    
    for var_name in self.required_vars:
      setattr(self, var_name, getattr(param_file, var_name))
    for var_name in self.optional_vars:
      setattr(self, var_name, getattr(param_file, var_name, None))
    
    if self.train_loss_check_every is None:
      self.train_loss_check_every = self.check_every
    
    if self.decay_function is not None:
      setattr(self, 'decay_params', getattr(param_file, 'decay_params'))
        
    if self.original_model is not None:
      setattr(self, 'up_to_layer', getattr(param_file, 'up_to_layer'))
      setattr(self, 'original_run', getattr(param_file, 'original_run', None))
      original_file = __import__(self.original_model)
      original_layers = getattr(original_file, 'layers')
      self.layers = original_layers[:(self.up_to_layer+1)] + self.layers
  
  def make_weights_path(self, ver='BEST'):
    ver = ver.upper()
    assert ver in ['BEST', 'LAST']
    return './Weights/{n}/Run_{r}/{n}_Run_{r}_{v}'.format(
                                                n=self.name, r=self.run, v=ver)

  def make_logs_path(self):
    return './Logs/{n}/Run_{r}'.format(n=self.name, r=self.run)

class CNNClassifier(object):
  """
  A CNNClassifier-class object can define and train a convolutional 
  neural network as specified by the attributes of its Parameters-class
  input.
  """
  def __init__(self, params):
    self.params = params
    
  def add_placeholders(self):
    """
    Creates placeholders including supplementary ones for layers 
    having different train- and test-behaviour, such as dropout and
    batch-normalization.
    """
    self.input_placeholder = tf.placeholder(tf.float32,
                                shape = [None] + self.params.image_dim,
                                name = 'Images')
    self.labels_placeholder = tf.placeholder(tf.float32,
                                shape = [None, self.params.n_classes],
                                name = 'Labels')
    self.is_training_placeholder = tf.placeholder_with_default(False,
                                shape = [],
                                name = 'Is_Training')
    self.dropout_placeholder = tf.placeholder_with_default(1.,
                                shape = [],
                                name = 'Dropout_Rate')
    self.extra_holders = {'is_training': self.is_training_placeholder,
                          'keep_prob': self.dropout_placeholder}
    
  def add_body(self, x):
    self.layer_outputs = []
    for i, layer in enumerate(self.params.layers):
      x = layer(x, self.extra_holders, 'Layer_{:d}'.format(i))
      self.layer_outputs.append(x)
    return x
  
  def add_loss(self, preds, eps=1e-15):
    """
    Takes predictions from model layers and returns cross-entropy loss,
    classification accuracy and final scores for each label.
    
    Optional input eps is added to predictions for numerical stability.
    """
    y_ = self.labels_placeholder
    loss = -tf.reduce_mean(tf.log(tf.reduce_sum(y_*preds,1)+eps))
    correct = tf.equal(tf.argmax(preds,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return loss, accuracy
    
  def add_training_op(self, loss):
    """
    Adds training-step operation to the computational graph. Optionally
    applies learning-rate decay and/or gradient clipping, if their
    required parameters have been set.
    """
    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    if self.params.decay_function == tf.train.piecewise_constant:
      self.lr = self.params.decay_function(
                          self.global_step, *self.params.decay_params)
    elif self.params.decay_function is not None:
      self.lr = self.params.decay_function(
                          self.params.learning_rate, self.global_step,
                          *self.params.decay_params)
    else:
      self.lr = self.params.learning_rate
    optim = self.params.optimizer(self.lr)
    grads_and_vars = optim.compute_gradients(loss)
    if self.params.clip_min_and_max is not None:
      def clipper(gv):
        clip = tf.clip_by_value(gv[0], *self.params.clip_min_and_max)
        return (clip, gv[1])
      grads_and_vars = [clipper(gv) for gv in grads_and_vars]
    return optim.apply_gradients(grads_and_vars, global_step=self.global_step)
    
  def build_model(self):
    self.add_placeholders()
    self.predictions = self.add_body(self.input_placeholder)
    self.loss, self.accuracy = self.add_loss(self.predictions)
    self.train_op = self.add_training_op(self.loss)
  
  def train_model(self, resume=False):
    """
    Trains the CNN Classifier and saves best validation-loss weights in
    ./Weights folder under the model's name, and the weights from the
    final iteration under the modelname + '_LAST'.
    
    tf.summary records of progress are saved in ./Logs under the model
    name, from where they can be viewed with TensorBoard.
    
    If optional argument 'resume' is True, training restarts from
    weights saved previously for the given run. In this case, wall-time
    plots in TensorBoard include time between the sessions, not just 
    the actual training time.
    """
    start_time = time.time()
    data = self.params.data_wrap()
        
    print('Building Computational Graph')
    g = tf.Graph()
    with g.as_default(), tf.Session() as session:
      with tf.variable_scope(self.params.name):
        self.build_model()
      variables = g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                   scope=self.params.name)
      saver = tf.train.Saver(variables)
      if resume:
        if self.params.weights_path[-4:] == 'LAST':
          print('Warning: Continuing training from a LAST checkpoint'
                ' will overwrite saved BEST')
        saver.restore(session, self.params.weights_path)
        self.params.weights_path = self.params.make_weights_path()
      else:
        session.run(tf.variables_initializer(variables))
        if self.params.run_was_set:
          print('Training (without resuming) will overwrite saved'
                ' {:s} Run {:d}'.format(self.params.name, self.params.run))
          self.params.weights_path = self.params.make_weights_path()
        else:
          self.params.run += 1
          self.params.weights_path = self.params.make_weights_path()
          self.params.logs_path = self.params.make_logs_path()
          os.makedirs(self.params.weights_path.rpartition('/')[0])
            
      train_writer = tf.summary.FileWriter(
                           sequential_name(self.params.logs_path,'train'))
      valid_writer = tf.summary.FileWriter(
                           sequential_name(self.params.logs_path,'validation'))
      test_writer = tf.summary.FileWriter(
                           sequential_name(self.params.logs_path,'test'))
      
      mean_sum = tf.reduce_mean(tf.reduce_sum(self.predictions, 1))
      main_summaries = [tf.summary.scalar('Loss', self.loss),
                        tf.summary.scalar('Accuracy', self.accuracy),
                        tf.summary.scalar('Mean_Sum_Of_Predictions', mean_sum)]
      lr_summary = [tf.summary.scalar('Learning_Rate', self.lr)]
      
      weights_and_biases = g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            scope=self.params.name+'/Layer_')
      wb_summaries = []
      for var in weights_and_biases:
        wb_summaries.append(tf.summary.histogram(var.op.name, var))
      
      dead_summaries = []
      if self.params.layers_to_track_dead_units is not None:
        for i in self.params.layers_to_track_dead_units:
          dead = tf.less_equal(tf.count_nonzero(self.layer_outputs[i], 0), 0)
          n_ = tf.reduce_mean(tf.cast(dead, tf.float32))*100.
          s_ = tf.summary.scalar('Percent_Inactive_Layer_{:d}'.format(i), n_)
          dead_summaries.append(s_)
      
      train_summaries = tf.summary.merge(main_summaries 
                                         + lr_summary)
      valid_summaries = tf.summary.merge(main_summaries 
                                         + wb_summaries 
                                         + dead_summaries)
      test_summaries = tf.summary.merge(main_summaries)
      
      valid_feed = {self.input_placeholder: data.validation.images,
                    self.labels_placeholder: data.validation.labels}
      train_feed = {self.is_training_placeholder: True,
                    self.dropout_placeholder: self.params.dropout_prob}
      
      if resume:
        best_acc = session.run(self.accuracy, feed_dict=valid_feed)
      else:
        best_acc = 0.
      best_step = 0
      
      batch_time = time.time()
      print('Training Model')
      for i in sm.range(self.params.n_batches):
        batch = data.train.next_batch(self.params.batch_size)
        train_feed[self.input_placeholder] = batch[0]
        train_feed[self.labels_placeholder] = batch[1]
        if (i+1)%self.params.train_loss_check_every == 0:
          train_summ, steps, _ = session.run([train_summaries, 
                                              self.global_step, 
                                              self.train_op], 
                                             feed_dict=train_feed)
          train_writer.add_summary(train_summ, steps)
        else:
          _ = session.run(self.train_op, feed_dict=train_feed)
        if (i+1)%self.params.check_every == 0:
          valid_summ, steps, loss, acc = session.run([valid_summaries, 
                                                      self.global_step,
                                                      self.loss, self.accuracy],
                                                     feed_dict=valid_feed)
          valid_writer.add_summary(valid_summ, steps)
          saver.save(session, self.params.make_weights_path('LAST'))
          print('batch {:d}: loss {:.5f}, accuracy {:.4f}, {:.2f} sec'
                               .format(i+1, loss, acc, time.time()-batch_time))
          batch_time = time.time()
          if acc > best_acc:
            best_acc = acc
            best_step = i+1
            saver.save(session, self.params.weights_path)
          if self.params.early_stopping is not None:
            if (i+1-best_step) >= self.params.early_stopping:
              break
      
      saver.save(session, self.params.make_weights_path('LAST'))
      
      if self.params.check_every <= self.params.n_batches:
        saver.restore(session, self.params.weights_path)
        best_step = session.run(self.global_step)
        print('Testing Model: Best Step = {:g}'.format(best_step))
        feed = {self.input_placeholder: data.test.images,
                self.labels_placeholder: data.test.labels}
        test_summ, steps, loss, acc = session.run([test_summaries, 
                                                   self.global_step,
                                                   self.loss, self.accuracy],
                                                  feed_dict=feed)
        test_writer.add_summary(test_summ, steps)
        print('test loss {:.5f}, accuracy {:.4f}'.format(loss, acc))
      
      train_writer.close()
      valid_writer.close()
      test_writer.close()
      
    print('Total Time {:.2f}'.format(time.time()-start_time))
    
    
def load_cnn(session, name, run=None):
  """
  Loads a previously-trained CNNClassifier object with the given name.
  The operations of that object are added to the current default graph,
  so load_cnn should not be called twice with the same name in the same
  graph. The trained variables are initialized in input tf.Session 
  'session'.
  """
  g = tf.get_default_graph()
  with g.as_default():
    params = Parameters(name, run)
    CNN = CNNClassifier(params)
    with tf.variable_scope(name):
      CNN.build_model()
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                  scope=name+'/')
    saver = tf.train.Saver(variables)
    saver.restore(session, params.weights_path)
  return CNN
  
