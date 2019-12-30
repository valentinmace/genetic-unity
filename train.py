# Valentin Macé
# valentin.mace@kedgebs.com
# Developed for fun
# Feel free to use this code as you wish as long as you quote me as author

"""
train.py
~~~~~~~~~~

Example code for training neural networks using a genetic algorithm

"""

from mlagents_envs.environment import UnityEnvironment

from genetic_algorithm import *

if __name__ == '__main__':

    """
    Path to Unity game
    
    I use a multi-agents environment to speed un training
    """
    multiple_agents_env_name = "./examples/builds/3DBall/multiple_agents/ball.exe"

    """
    Train your own neural networks

    You can start the genetic algorithm as follow
    I tested these hyperparameters with good results
    
    You might want to use multi-processing, but note that it is still an early version that might not work properly
    Use multiprocessing.cpu_count() to count available cores
    """
    gen = GeneticAlgorithm(unity_env_name=multiple_agents_env_name, networks_shape=[8,16,2], population_size=100,
                           crossover_method='neuron', mutation_method='weight', n_process=4)
    gen.start()
