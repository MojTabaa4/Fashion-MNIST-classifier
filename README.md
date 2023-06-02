# Fashion MNIST Classifier

## Introduction

In this repository, I have implemented a feedforward neural network for classifying the Fashion-MNIST dataset using only
the NumPy library. The implementation is object-oriented and includes classes for the data loader, activation functions,
loss functions, neural layers, and the feedforward neural network itself.

I have used pure NumPy library to implement the neural network. my implementation is object-oriented,
with classes to define the data loader, activation functions, loss functions, neural layers, and the feedforward neural
network itself.

## Dataset

The Fashion-MNIST dataset is a well-known benchmark dataset in the machine learning community. It was created as a
drop-in replacement for the MNIST dataset, which had become too easy for modern machine learning algorithms. The
Fashion-MNIST dataset is more challenging and realistic because it contains images of clothing items instead of
handwritten digits.

The dataset consists of 70,000 images of 28x28 pixels. There are 60,000 training images and 10,000 test images. Each
image represents one of 10 different classes of clothing items, including T-shirts/tops, trousers, pullovers, dresses,
coats, sandals, shirts, sneakers, bags, and ankle boots.

## Data Loader

The `DataLoader` class is responsible for loading the dataset into memory and providing batches of data during training
and testing. It takes the data and labels arrays, batch size, and shuffle flag as input. The `__iter__` method returns a
tuple of numpy arrays of size `(batch_size, image_size)` and `(batch_size, )`, representing a batch of images and their
corresponding labels, respectively.

## Activation Functions

The `ActivationFunction` is an abstract class that defines the interface for the activation functions used in the neural
network. The `__compute_value` method computes the activation function value for a given input matrix, and
the `derivative` method computes the derivative of the activation function. The `__call__` method calls
the `__compute_value` method.

The following activation functions are implemented as subclasses of `ActivationFunction`:

- ### Identical

  The `Identical` activation function returns the input matrix as is. It is used as the last activation function in the
  neural network to output the predicted class probabilities.

- ### Relu

  The `Relu` activation function returns the max of 0 and the input matrix. It is commonly used as an activation
  function
  in neural networks because it is fast to compute and prevents the vanishing gradient problem.

- ### LeakyRelu

  The `LeakyRelu` activation function returns the max of 0 and the input matrix plus a small negative slope. It is a
  variant of the `Relu` activation function that prevents the dying ReLU problem by allowing a small gradient for
  negative
  input values.

- ### Sigmoid

  The `Sigmoid` activation function returns the sigmoid of the input matrix. It is commonly used as an activation
  function
  in binary classification problems because it maps the input values to a probability between 0 and 1.

- ### Softmax
  The `Softmax` activation function returns the softmax of the input matrix. It is used to output the predicted class
  probabilities in multi-class classification problems by normalizing the output values to sum up to 1.

## Cross Entropy Loss

The `CrossEntropy` class computes the cross-entropy loss between the true labels and the predicted class probabilities.
It has a `__compute_value` method to compute the loss value and a `derivative` method to compute the derivative of the
loss function. The cross-entropy loss is a common choice for multi-class classification problems because it penalizes
incorrect predictions more heavily than correct predictions.

## Neural Layer

The `NeuralLayer` class defines a single layer of the neural network. It takes the input size, number of neurons,
activation function, and weight initialization method as input. It has a `forward` method to compute the output of the
layer given an input and a `update_weights` method to update the layer weights during backpropagation. The weights are
initialized using the specified weight initialization method, which can be either random or zero.

## Feed Forward NN

The `FeedForwardNN` class defines the feedforward neural network. It takes the input shape as input and has methods to
add layers, set the training parameters, perform forward pass, and train the network. The `fit` method trains the
network for a given number of epochs using the training data and optionally evaluates the network on the test data after
each epoch. The `forward` method performs the forward pass through the entire network. The `add_layer` method adds a new
layer to the network with the specified number of neurons, activation function, and weight initialization method.
The `set_training_param` method sets the loss function and learning rate for the network.

## Training

The code block initializes a feedforward neural network, adds four layers with 20 neurons each, sets the training
parameters, and trains the network using the Fashion-MNIST dataset. The number of epochs is set to 25, and the batch
size is set to 32. The `fit` method returns a log of the training and test losses and accuracies for each epoch. The
training and test losses and accuracies are plotted using Matplotlib.

The training process involves adjusting the weights of the neural network using backpropagation, a method for computing
the gradients of the loss function with respect to the weights. The gradients are used to update the weights in the
direction that minimizes the loss function. The learning rate controls the step size of the weight updates, and the
batch size determines the number of samples used to compute each gradient update.

## Conclusion

In summary, I have implemented a feedforward neural network to classify the Fashion-MNIST dataset using TensorFlow and
NumPy libraries. I have defined classes for the data loader, activation functions, loss functions, neural layers, and
the neural network itself. I have trained the network using backpropagation and evaluated its performance using the
test data. The implementation can be extended and customized for other machine learning problems.
