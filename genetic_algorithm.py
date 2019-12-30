# Valentin Macé
# valentin.mace@kedgebs.com
# Developed for fun
# Feel free to use this code as you wish as long as you quote me as author

"""
genetic_algorithm.py
~~~~~~~~~~

A module to implement a genetic algorithm to train neural networks in a Unity Environment with ml-agents

"""

import time
from random import randint
from game import*
from mlagents_envs.environment import UnityEnvironment

from crossover import *
from evaluation import *
from mutation import *
from neural_network import *


class GeneticAlgorithm:
    """ Genetic Algorithm Class """

    def __init__(self, unity_env_name, networks=None, networks_shape=None, population_size=1000, generation_number = 100,
                 crossover_rate=0.3, crossover_method='neuron', mutation_rate=0.7, mutation_method='weight', n_process=1):
        """ Initializes the genetic algorithm

        :param networks(list of NeuralNetwork): First generation networks
        :param networks_shape(list of int): List defining number of layers and number of neurons in each layer
        :param population_size(int): Number of networks for each generation
        :param generation_number(int): How many generations the algorithm will run
        :param crossover_rate(int): Proportion of children to be produced at each generation
        :param crossover_method(str): How children will be produced
        :param mutation_rate(int): Proportion of the population to mutate at each generation
        :param mutation_method(str): How mutation will be done

        Todo: Replace n_agent as it will describe the number of simulations
        Todo: use yaml for easier configuration,
        Todo: automate NeuralNetwork inner and outer shape (let hidden layers as free choice)
        """
        self.networks_shape = networks_shape
        if self.networks_shape is None:             # if no shape is provided
            self.networks_shape = [21,16,3]         # default shape
        self.networks = networks

        if networks is None:                                  # if no networks are provided
            self.networks = []
            for i in range(population_size):                  # producing population
                self.networks.append(NeuralNetwork(self.networks_shape))

        self.population_size = population_size
        self.generation_number = generation_number
        self.crossover_rate = crossover_rate
        self.crossover_method = crossover_method
        self.mutation_rate = mutation_rate
        self.mutation_method = mutation_method
        self.n_process = n_process

        self.unity_env_name = unity_env_name
        self.env = UnityEnvironment(base_port = 5006, file_name=unity_env_name, seed=0, no_graphics=True)
        self.env.reset()
        group_name = self.env.get_agent_groups()[0]
        self.n_agents = self.env.get_step_result(group_name).n_agents()

        self.generation_durations = []
        self.generation_performances = []

    def start(self):
        """
        Main function operating the Genetic Algorithm

        Steps at each generation:
        1- Parents selection
        2- Offsprings production
        3- Mutated individuals production
        4- Evaluation of whole population (old population + offsprings + mutated individuals)
        5- Additional mutations on random individuals (seems to improve learning)
        6- Keeping only population_size individuals, throwing bad performers

        Todo: Consider different sequences of steps, make it modular or user built
        """
        networks = self.networks
        population_size = self.population_size
        crossover_number = int(self.crossover_rate*self.population_size)   # calculate number of children to be produced
        mutation_number = int(self.mutation_rate*self.population_size)     # calculate number of mutation to be done

        gen = 0                                         # current generation
        for _ in range(self.generation_number):
            start_time = time.time()
            gen += 1

            parents = self.parent_selection(networks, crossover_number, population_size)       # parent selection
            children = self.children_production(crossover_number, parents)                     # children making
            mutations = self.mutation_production(networks, mutation_number, population_size)   # mutations making

            networks = networks + children + mutations                      # old population and new individuals
            self.evaluation(networks)                                       # evaluation of neural nets
            networks.sort(key=lambda Network: Network.score, reverse=True)  # ranking neural nets
            networks[0].save(name="gen_"+str(gen))                          # saving best of current generation

            for _ in range(int(0.2*len(networks))):              # More random mutations because it helps
                rand = randint(10, len(networks)-1)
                networks[rand] = mutation(networks[rand], self.mutation_method)

            networks = networks[:population_size]       # Keeping only best individuals
            end_time = time.time()
            iteration_time = end_time-start_time
            self.print_generation(networks, gen, iteration_time)

    def parent_selection(self, networks, crossover_number, population_size):
        """
        Parent selection function, takes 3 random individuals and makes a tournament between them,
        the winner is selected as a parent

        :param networks:(list of NeuralNetwork) Neural nets to be selected as parents or not
        :param crossover_number:(int) Number of parents needed
        :param population_size:(int) Size of whole neural nets population
        :return:(list of NeuralNetwork) list of selected parents

        Todo: Use pseudo parallelization to speed up
        """
        parents = []
        for i in range(crossover_number):
            parent = self.tournament(networks[randint(0, population_size - 1)],
                                     networks[randint(0, population_size - 1)],
                                     networks[randint(0, population_size - 1)])
            parents.append(parent)
        return parents

    def children_production(self, crossover_number, parents):
        """
        Takes randomly 2 parents in the parents list and makes them crossover to give a child
        Note: the crossover method is contained in self.crossover_method

        :param crossover_number:(int) Number of children needed
        :param parents:(list of NeuralNetwork) Potential parents
        :return:(list of NeuralNetwork) Children

        Todo: Use pseudo parallelization to speed up
        """
        children = []
        for i in range(crossover_number):
            child = crossover(self.env, parents[randint(0, crossover_number - 1)],
                              parents[randint(0, crossover_number - 1)], self.crossover_method)
            children.append(child)
        return children

    def mutation_production(self, networks, mutation_number, population_size):
        """
        Makes new individuals from individuals in the current population by mutating them
        Note: it does not affect current individuals but actually creates new ones

        :param networks:(list of NeuralNetwork) Neural nets to randomly become mutants
        :param mutation_number:(int) number of mutants needed
        :param population_size:(int) Size of whole neural nets population
        :return:(list of NeuralNetwork) New individuals (mutants)

        Todo: Use pseudo parallelization to speed up
        """
        mutants = []
        for i in range(mutation_number):
            mut = mutation(networks[randint(0, population_size - 1)], self.mutation_method)      # mutant making
            mutants.append(mut)                                               # append mutant
        return mutants

    def evaluation(self, networks):
        """
        Takes the population of neural nets and makes them play 4 games each, a NeuralNetwork.score is the mean
        of its 4 games

        :param networks:(list of NeuralNetwork)
        """

        if self.n_process == 1:
            results = single_process_evaluation(self.env, networks, self.n_agents)
            for i in range(len(networks)):
                networks[i].score = np.mean([results[0][i], results[1][i], results[2][i], results[3][i]])
        else:
            results = multi_process_evaluation(networks, self.unity_env_name, self.n_process, self.n_agents)
            for i in range(len(networks)):
                networks[i].score = results[i]

    def tournament(self, net1, net2, net3):
        """
        Takes 3 neural nets, makes them play a game each and select the best performer

        :param net1:(NeuralNetwork) 1st participant
        :param net2:(NeuralNetwork) 2nd participant
        :param net3:(NeuralNetwork) Last but not least, the third contender
        :return:(NeuralNetwork) The winning neural net

        Todo: Use pseudo parallelization to speed up
        Todo: Write better return conditions
        """
        game = Game(unity_env=self.env, time_scale=100.0, width=0, height=0, target_frame_rate=-1, quality_level=0)
        score1 = game.start([net1])                # net1 plays a game and so on..
        score2 = game.start([net2])
        score3 = game.start([net3])
        maxscore = max(score1, score2, score3)     # the best one is returned
        if maxscore == score1:
            return net1
        elif maxscore == score2:
            return net2
        else:
            return net3

    def print_generation(self, networks, gen, iteration_time):
        """
        Shows info about current generation

        Todo: Replace with more complete performance analysis, plots etc. Store results in a file maybe
        """
        self.generation_durations.append(iteration_time)
        self.generation_performances.append(np.mean([networks[i].score for i in range(len(networks))]))
        top_mean = np.mean([networks[i].score for i in range(6)])
        bottom_mean = np.mean([networks[-i].score for i in range(1, 6)])
        print("Pop size = ", len(networks))
        print("\nDuration : ", iteration_time)
        print("Best Fitness gen", gen, " : ", networks[0].score)
        print("Average all = ", self.generation_performances[-1])
        print("Average top 6 = ", top_mean)
        print("Average last 6 = ", bottom_mean)

