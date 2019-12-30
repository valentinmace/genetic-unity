# Valentin Macé
# valentin.mace@kedgebs.com
# Developed for fun
# Feel free to use this code as you wish as long as you quote me as author

"""
crossover.py
~~~~~~~~~~

A module to implement all crossover routines used in a genetic algorithm

"""

import copy
from random import randint

from game import *


def crossover(env, parent1, parent2, crossover_method):
    """
    Takes two neural nets and produce a child according to :param crossover_method
    Example of working (method = 'neuron'):
    1- Two networks are created (copies of each parent)
    2- Selects a random neuron in a random layer OR a random bias in a random layer
    3- Switches this neuron OR bias between the two networks
    4- Each network plays a game
    5- Best one is selected
    Principle is the same for weight or layer methods

    :param env:(UnityEnvironment) Environment where evaluation games will be played
    :param parent1:(NeuralNetwork) first parent
    :param parent2:(NeuralNetwork) second parent
    :param crossover_method:(str) to apply crossover over a single weight, a neuron or an entire layer
    :return:(NeuralNetwork) Child
    """
    net1 = copy.deepcopy(parent1)  # making copies (children) otherwise we manipulate the actual parents
    net2 = copy.deepcopy(parent2)
    weights_or_biases = randint(0, 1)
    if weights_or_biases == 0:
        if crossover_method == 'weight':
            weight_crossover(net1, net2)
        elif crossover_method == 'neuron':
            neuron_crossover(net1, net2)
        elif crossover_method == 'layer':
            layer_crossover(net1, net2)
    else:  # crossover over bias
        bias_crossover(net1, net2)

    game = Game(unity_env=env, time_scale=100.0, width=0, height=0, target_frame_rate=-1, quality_level=0)
    score1 = game.start([net1])
    score2 = game.start([net2])
    if score1 > score2:
        return net1
    else:
        return net2


def weight_crossover(net1, net2):
    """
    Switches a single weight between two NeuralNetwork

    :param net1:(NeuralNetwork) First parent
    :param net2:(NeuralNetwork) Second parent
    """
    layer = randint(0, len(net1.weights) - 1)  # random layer
    neuron = randint(0, len(net1.weights[layer]) - 1)  # random neuron
    weight = randint(0, len(net1.weights[layer][neuron]) - 1)  # random weight
    temp = net1.weights[layer][neuron][weight]  # switching weights
    net1.weights[layer][neuron][weight] = net2.weights[layer][neuron][weight]
    net2.weights[layer][neuron][weight] = temp


def neuron_crossover(net1, net2):
    """
    Switches neuron between two NeuralNetwork

    :param net1:(NeuralNetwork) First parent
    :param net2:(NeuralNetwork) Second parent
    """
    layer = randint(0, len(net1.weights) - 1)  # random layer
    neuron = randint(0, len(net1.weights[layer]) - 1)  # random neuron
    temp = copy.deepcopy(net1)  # switching neurons
    net1.weights[layer][neuron] = net2.weights[layer][neuron]
    net2.weights[layer][neuron] = temp.weights[layer][neuron]


def layer_crossover(net1, net2):
    """
    Switches a whole layer between two NeuralNetwork

    :param net1:(NeuralNetwork) First parent
    :param net2:(NeuralNetwork) Second parent
    """
    layer = randint(0, len(net1.weights) - 1)  # random layer
    temp = copy.deepcopy(net1)  # switching layers
    net1.weights[layer] = net2.weights[layer]
    net2.weights[layer] = temp.weights[layer]


def bias_crossover(net1, net2):
    """
    Switches a single bias between two NeuralNetwork

    :param net1: (NeuralNetwork) First parent
    :param net2: (NeuralNetwork) Second parent
    """
    layer = randint(0, len(net1.biases) - 1)  # random layer
    bias = randint(0, len(net1.biases[layer]) - 1)  # random bias
    temp = copy.deepcopy(net1)  # switching biases
    net1.biases[layer][bias] = net2.biases[layer][bias]
    net2.biases[layer][bias] = temp.biases[layer][bias]
