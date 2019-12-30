# Valentin Macé
# valentin.mace@kedgebs.com
# Developed for fun
# Feel free to use this code as you wish as long as you quote me as author

"""
neural_network.py
~~~~~~~~~~

This module is for building a classic dense neural network

"""

from numba import jit
import numpy as np


class NeuralNetwork:
    """ Neural Network class """

    def __init__(self, shape=None):
        """ Initializes the neural network

        Weights and biases are initialized randomly according to a normal distribution
        A network can be saved and loaded for later use

        :param shape:(list of int) Describes how many layers and neurons by layer the network has
        """
        self.shape = shape
        self.biases = []
        self.weights = []
        self.score = 0        # to remember how well it performed
        if shape:
            for y in shape[1:]:                             # biases random initialization
                self.biases.append(np.random.randn(y, 1))
            for x, y in zip(shape[:-1], shape[1:]):         # weights random initialization
                self.weights.append(np.random.randn(y, x))
        self.depth = len(self.biases)

    def feed_forward(self, a):
        """
        Main function, takes an input vector and calculate the output by propagation through the network

        :param a: column of integers, inputs for the network (snake's vision)
        :return: column of integers, output neurons activation
        """
        weights = self.weights
        biases = self.biases
        for i in range(self.depth):
            a = sigmoid(np.dot(weights[i], a) + biases[i])
        return a

    def save(self, name=None):
        """
        Saves network weights and biases into 2 separated files in current folder

        :param name: str, in case you want to name it
        :return: creates two files
        """
        if not name:
            np.save('saved_weights_'+str(self.score), self.weights)
            np.save('saved_biases_'+str(self.score), self.biases)
        else:
            np.save(name + '_weights', self.weights)
            np.save(name + '_biases', self.biases)

    def load(self, filename_weights, filename_biases):
        """
        Loads saved network weights and biases from 2 files into the actual network object

        :param filename_weights: file containing saved weights
        :param filename_biases: file containing saved biases
        """
        self.weights = np.load(filename_weights)
        self.biases = np.load(filename_biases)


@jit(nopython=True)
def sigmoid(z):
    """
    The sigmoid function, classic neural net activation function
    @jit is used to speed up computation
    """
    return 1.0 / (1.0 + np.exp(-z))
