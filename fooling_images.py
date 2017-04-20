from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import six.moves as sm
import tensorflow as tf

from cnn_classifier import Parameters, CNNClassifier, load_cnn
from utils import sequential_name


def make_plot(start_label, start_image, start_score,
              target_label, final_image, final_score,
              model_name, save_plot=True, show_plot=False):  
  start_image = np.squeeze(start_image)
  final_image = np.squeeze(final_image)
  plt.clf()
  plt.subplot(1, 2, 1)
  plt.imshow(start_image, cmap=plt.cm.gray)
  plt.axis('off')
  plt.title('Start Image: {:d} ({:.4f})'.format(start_label, start_score))
  plt.subplot(1, 2, 2)
  plt.imshow(final_image, cmap=plt.cm.gray)
  plt.axis('off')
  plt.title('Final Image: {:d} ({:.4f})'.format(target_label, final_score))
  plot_name = '{:s} {:d}-{:d}'.format(model_name, start_label, target_label)
  plot_name = sequential_name('./Fooling_Images', plot_name)
  if save_plot:
    plt.savefig(plot_name)
  if show_plot:
    plt.show()
  else:
    plt.clf()


class FoolingModel(object):
  """
  Class for creating fooling images against a CNNClassifier model
  """
  def __init__(self, modelname, run=None, min_valid=0., max_valid=1., lr=5e-4,
               optimizer=tf.train.GradientDescentOptimizer):
    """
    Defines the operations for creating fooling images based on the 
    input model.
    
    A 'fooling' variable-scope is used so that optimizers which create
    variables, such as Adam, can have those variables accessed and 
    re-initialized separately from the main model.
    
    Inputs:
    - modelname: name used in the CNN model to make fooling images from
    - run: optional, which of the model's training runs weights to use:
      can be int - number of the run, or list/tuple of that number and 
      'BEST' or 'LAST'
    - min_valid, max_valid: floats, minimum and maximum values allowed
                            in a valid image
    - lr: float, learning rate when making fooling images
    - optimizer: one of the tf.train.Optimizer classes
    """
    self.g = tf.Graph()
    
    with self.g.as_default():
      self.session = tf.Session()
      model = load_cnn(self.session, modelname, run)
      self.params = model.params
      image_shape = [1]+model.params.image_dim
      optim = optimizer(lr)
      self.input_placeholder = tf.placeholder(tf.float32,
                                            shape=image_shape,
                                            name='Start_Image')
      self.labels_placeholder = model.labels_placeholder
    
      self.initial_var = tf.Variable(np.zeros(image_shape, np.float32))
      x = self.initial_var+self.input_placeholder
      self.clipped = tf.clip_by_value(x, min_valid, max_valid)
      with tf.variable_scope(modelname, reuse=True):
        self.predictions = model.add_body(self.clipped)
        loss, self.accuracy = model.add_loss(self.predictions)
    
      with tf.variable_scope('fooling'):
        grad = optim.compute_gradients(loss, [self.initial_var])
        self.step = optim.apply_gradients(grad)
      self.fooling_vars = [self.initial_var] + self.g.get_collection(
                                                 tf.GraphKeys.GLOBAL_VARIABLES,
                                                 scope='fooling')
  
  def select_image(self, start_label, data, max_iter=1000):
    """
    Returns one test-set image of 'data' with true label 'start_label'
    that is correctly classified by the base model-to-be-fooled, as
    well as the score given by the model to that true label.
  
    Inputs:
    - start_label: int, ground truth label of image to select
    - data: dataset to choose image from
    """
    with self.g.as_default():
      indices = np.where(data.test.labels[:,start_label] == 1)[0]
      self.session.run(tf.variables_initializer([self.initial_var]))
      for i in sm.range(max_iter):
        choice = data.test.images[np.random.choice(indices),:][np.newaxis,]
        feed = {self.input_placeholder: choice}
        pred_labels = self.session.run(self.predictions, feed_dict=feed)
        if np.argmax(pred_labels) == start_label:
          print('starting image found ({:d})'.format(start_label))
          return choice, pred_labels[0,start_label]
      print('max iterations reached, no starting image found')
      return None, None
  
  def create_image(self, start_image, target_label,
                   min_score=0.9, print_every=100, max_iter=5000,
                   save_plot=True, show_plot=False):
    """
    Performs the optimization for creating a single fooling image.
    Stops when maximum iterations has been reached, or image is 
    classified as target label with a score of at least min_score.
    
    Inputs:
    - start_image: numpy array matching shape of images in the model to
                be fooled, the initial image to build from
    - target_label: int, which class the model should think the final
                image is from
    - min_score: float less than 1, minimum score of target class the 
                fooling image should achieve
    - print_every: int, updates between each report of progress
    - max_iter: int, maximum number of updates to perform
    - save_plot, show_plot: booleans, whether to save/show plot from
                make_plot after final image is created
    
    Returns:
    - final_image: numpy array, same shape as start_image
    - score: np.float, the score assigned by the model to target_label
                for final_image
    """
    label_onehot = np.zeros((1,self.params.n_classes))
    label_onehot[0,target_label] = 1.
    
    with self.g.as_default():
      feed = {self.input_placeholder: start_image,
              self.labels_placeholder: label_onehot}
      
      self.session.run(tf.variables_initializer(self.fooling_vars))
      
      scores = self.session.run(self.predictions, feed_dict=feed)
      start_label = np.argmax(scores)
      start_score = np.max(scores)
      
      for i in sm.range(max_iter):
        correct, scores, _ = self.session.run(
                  [self.accuracy, self.predictions, self.step], feed_dict=feed)
        score = scores[0,target_label]
        if correct == 1. and score > min_score:
          break
        if (i+1)%print_every == 0:
          print('iteration {:d}: score {:.4g}'.format(i+1, score))
      print('finished at {:d} iterations ({:d})'.format(i+1, target_label))
      feed = {self.input_placeholder: start_image}
      final_image = self.session.run(self.clipped, feed_dict=feed)
    if save_plot or show_plot:
      name_on_plot = self.params.weights_path.rpartition('/')[2]
      make_plot(start_label, start_image, start_score, target_label, 
                final_image, score, name_on_plot, save_plot, show_plot)
    return final_image, score

  
def fool_model(n_images, modelname, run=None, start_and_target=None, 
               max_search=1000, min_valid=0., max_valid=1., lr=5e-4,
               optimizer=tf.train.GradientDescentOptimizer,
               min_score=0.9, print_every=100, max_updates=5000,
               save_plot=True, show_plot=False):
  """
  Creates fooling images for pre-trained CNNClassifier created with
  name 'modelname'. The starting and target labels for the images can 
  be set with optional argument start_and_target. 
  The images are saved in folder ./Fooling_Images.
  """
  fooling_model = FoolingModel(modelname, run, min_valid, 
                               max_valid, lr, optimizer)
  data = fooling_model.params.data_wrap()
  for _ in sm.range(n_images):
    if start_and_target is None:
      start_label, target_label = np.random.choice(
                                      sm.range(fooling_model.params.n_classes),
                                      size=2, replace=False)
    else:
      start_label, target_label = start_and_target
    start_image, _ = fooling_model.select_image(start_label, data, max_search)
    if start_image is not None:
      final_image, final_score = fooling_model.create_image(
                                           start_image, target_label,
                                           min_score, print_every, max_updates,
                                           save_plot, show_plot)


