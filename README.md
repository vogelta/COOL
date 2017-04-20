## The Competitive Overcomplete Output Layer in TensorFlow

An accompanying discussion can be found [**here**](https://vogelta.github.io/COOL)

The [Competitive Overcomplete Output Layer](https://arxiv.org/abs/1609.02226) is an alternative to the classic Softmax output for neural network classifiers. This project contains code for implementing the COOL and training models in TensorFlow, for training transfer-learning models, and for creating [fooling images](https://arxiv.org/abs/1412.1897) based on those models.
      
Some pretrained models on the MNIST dataset are included with their parameters, weights and logs.

### Prerequisites

- Python 2.7.12 or Python 3.5+
- TensorFlow 0.12.0
- NumPy 1.11.3
- Six 1.10.0
- Matplotlib 1.5.1

### Credits

- [Kardan & Stanley 'COOL' original paper](https://arxiv.org/abs/1609.02226)
- ['LeNet' network structure](http://yann.lecun.com/exdb/lenet/)
- [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow MNIST dataset loader](https://www.tensorflow.org/get_started/mnist/pros#load_mnist_data)

### Trained Models

Includes Parameters, Weights (highest-scoring-step weights only) and Logs for: <br>
(test-set accuracy in parentheses)

- 'MNIST_Softmax' - (0.9930)
- 'MNIST_COOL' - (0.9942)
- 'MNIST_MinCOOL' - (0.9938)

Full-size models take about 65 minutes on Intel® Core™ i5-2500K @ 3.30GHz. <br> 
Transfer-learning models take about 4 minutes on the same.
