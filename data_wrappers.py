from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six.moves as sm
import tensorflow as tf
import cPickle
import os

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from utils import one_hot


class DataSet_(object):
  """
  Based on tensorflow.contrib.learn.python.learn.datasets.mnist,
  specifically the DataSet class - removed parts assuming use of the
  MNIST dataset, and now always keeps shape/type of inputs.
  """
  def __init__(self, images, labels):
    assert images.shape[0] == labels.shape[0]
    self._num_examples = images.shape[0]
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """
    Returns a (images, labels)-tuple batch of batch_size examples.
    Handles shuffling the examples for each new epoch itself (the first
    epoch is passed in the original order).
    """
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


class OverallDataSet(object):
  """
  Based on the dataset used in Tensorflow MNIST tutorial. In that case
  a collections.namedtuple() is actually used for the equivalent role 
  of holding DataSet objects.
  Optional input preprocess_fn can be included to allow for handling
  new examples in the original, non-preprocessed format.
  Assumes there is sufficient memory available for the entire dataset.
  """
  def __init__(self, train_examples, train_labels,
               validation_examples, validation_labels,
               test_examples, test_labels,
               preprocess_fn=None):
    self.train = DataSet_(train_examples, train_labels)
    self.validation = DataSet_(validation_examples, validation_labels)
    self.test = DataSet_(test_examples, test_labels)
    if preprocess_fn is not None:
      self.preprocess_fn = preprocess_fn

def mnist_wrapper():
  mnist = read_data_sets('./Datasets/MNIST_data', one_hot=True)
  shape = [-1,28,28,1]
  mnist.train._images = mnist.train._images.reshape(shape)
  mnist.validation._images = mnist.validation._images.reshape(shape)
  mnist.test._images = mnist.test._images.reshape(shape)
  return mnist

def cifar_wrapper():
  """
  Makes a wrapper for CIFAR-10 dataset. Assumes the data is already
  downloaded and placed in ./Datasets/CIFAR_data/cifar-10-batches-py
  folder (as the python/Pickle version).
  
  The CIFAR data is at https://www.cs.toronto.edu/~kriz/cifar.html
  """
  folder = './Datasets/CIFAR_data/cifar-10-batches-py'
  train_data = []
  train_labels = []
  for i in sm.range(1,6):
    file_name = folder+'/data_batch_{:d}'.format(i)
    f = open(file_name, 'rb')
    d = cPickle.load(f)
    train_data.append(d['data'])
    train_labels.append(one_hot(d['labels'], 10))
    f.close()
  
  train_data = np.vstack(train_data)
  train_labels = np.vstack(train_labels)
  
  file_name = folder+'/test_batch'
  f = open(file_name, 'rb')
  d = cPickle.load(f)
  test_data = d['data']
  test_labels = one_hot(d['labels'],10)
  f.close()
  
  validation_size = 5000
  validation_data = train_data[-validation_size:,:]
  validation_labels = train_labels[-validation_size:,:]
  train_data = train_data[:-validation_size,:]
  train_labels = train_labels[:-validation_size,:]
  
  m = np.mean(train_data, axis=0, keepdims=True)
  s = np.std(train_data, axis=0, keepdims=True)
  s = np.fmax(s, np.ones_like(s)/3072.)
  
  def cifar_preprocess(data):
    data = (data-m)/s
    data = data.reshape([-1,3,32,32])
    return data.transpose([0,2,3,1])
  train_data, validation_data, test_data = [cifar_preprocess(d) for d in
                                      [train_data, validation_data, test_data]]
  
  cifar = OverallDataSet(train_data, train_labels,
                         validation_data, validation_labels, 
                         test_data, test_labels,
                         cifar_preprocess)
  return cifar

