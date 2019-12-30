# Valentin Macé
# valentin.mace@kedgebs.com
# Developed for fun
# Feel free to use this code as you wish as long as you quote me as author

"""
mutation.py
~~~~~~~~~~

A module to implement all mutation routines used in a genetic algorithm

"""

import copy
import numpy as np
from random import randint


def mutation(net, mutation_method):
    """
    Takes a NeuralNetwork and makes a clone with a mutation according to :param mutation_method

    :param net:(NeuralNetwork) Neural net subject to mutation (will be cloned)
    :param mutation_method:(str) Where to apply mutation (weights, neuron ...)
    :return:(NeuralNetwork) Cloned NeuralNetwork with the mutation
    """
    res = copy.deepcopy(net)  # making copy otherwise we manipulate the actual net in params
    weights_or_biases = randint(0, 1)
    if weights_or_biases == 0:
        if mutation_method == 'weight':
            weight_mutation(res)
        elif mutation_method == 'neuron':
            neuron_mutation(res)
    else:   # mutation over bias
        bias_mutation(res)
    return res


def weight_mutation(net):
    """
    Applies mutation to a NeuralNetwork's random weight

    :param net:(NeuralNetwork) Mutant
    """
    layer = randint(0, len(net.weights) - 1)  # random layer
    neuron = randint(0, len(net.weights[layer]) - 1)  # random neuron
    weight = randint(0, len(net.weights[layer][neuron]) - 1)  # random weight
    net.weights[layer][neuron][weight] = np.random.randn()  # mutation


def neuron_mutation(net):
    """
    Applies mutation to a NeuralNetwork's random neuron

    :param net:(NeuralNetwork) Mutant
    """
    layer = randint(0, len(net.weights) - 1)  # same logic here
    neuron = randint(0, len(net.weights[layer]) - 1)
    new_neuron = np.random.randn(len(net.weights[layer][neuron]))
    net.weights[layer][neuron] = new_neuron


def bias_mutation(net):
    """
    Applies mutation to a NeuralNetwork's random bias

    :param net:(NeuralNetwork) Mutant
    """
    layer = randint(0, len(net.biases) - 1)  # random layer
    bias = randint(0, len(net.biases[layer]) - 1)  # random bias
    net.weights[layer][bias] = np.random.randn()  # mutation