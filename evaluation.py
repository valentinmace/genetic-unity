# Valentin Macé
# valentin.mace@kedgebs.com
# Developed for fun
# Feel free to use this code as you wish as long as you quote me as author

"""
evaluation.py
~~~~~~~~~~

A module to implement all evaluation routines used in a genetic algorithm

"""

import multiprocessing as mp
import numpy as np
from mlagents_envs.environment import UnityEnvironment

from game import *


def single_process_evaluation(env, networks, n_agents):
    """
    Evaluate all networks over 4 games

    :param env:(UnityEnvironment) Environment where evaluation games will be played
    :param networks:(list of NeuralNetwork) Neural nets to be evaluated
    :param n_agents:(int) Number of agents in env, so we can pass multiple NeuralNetwork
    :return:(list of float) Scores for each NeuralNetwork

    Todo: Randomize env seed, average results here before return
    """
    game = Game(unity_env=env, time_scale=100.0, width=0, height=0, target_frame_rate=-1, quality_level=0)
    results = []
    for _ in range(4):
        sub_results = []
        for i in range(0, len(networks), n_agents):
            sub_results += game.start(networks[i:i + n_agents])
        results.append(sub_results)
    return results


def multi_process_evaluation(networks, env_name, n_process, n_agents):
    """
    Manages evaluation of all networks over multiple process

    :param networks:(list of NeuralNetwork) Neural nets to be evaluated
    :param env_name:(str) Path to built unity game
    :param n_process:(int) Number of process needed for parallelization
    :param n_agents:(int) Number of agents in the env, so we can pass multiple NeuralNetwork
    :return:(list of float) Scores for each NeuralNetwork

    Todo: Optimize, write clearer code
    """
    queue = mp.Queue()
    split_networks = np.array_split(networks, n_process)
    jobs = [mp.Process(target=multi_process_evaluation_job, args=(queue, split_networks[i], env_name, i, n_agents))
            for i in range(n_process)]
    for job in jobs: job.start()
    sub_results = [queue.get() for _ in range(n_process)]
    for job in jobs: job.join()
    sub_results.sort(key=lambda tup: tup[0])
    results = []
    for sub_result in sub_results: results += np.mean(sub_result[1], axis=0).tolist()
    return results


def multi_process_evaluation_job(queue, networks, env_name, worker, n_agents):
    """
    Single process evaluating its neural networks

    :param queue:(multiprocessing.Queue) Where to put results for the parent process
    :param networks:(list of NeuralNetwork) Neural nets to be evaluated
    :param env_name:(str) Path to built unity game
    :param worker:(int) Will be added to base_port, in order to use another port than existing UnityEnvironment
    :param n_agents:(int) Number of agents in the env, so we can pass multiple NeuralNetwork
    """
    env = UnityEnvironment(base_port=5006, worker_id=worker+1, file_name=env_name, seed=np.random.randint(0,100), no_graphics=True)
    game = Game(unity_env=env, time_scale=100.0, width=0, height=0, target_frame_rate=-1, quality_level=0)
    results = []
    for _ in range(4):
        sub_results = []
        for i in range(0, len(networks), n_agents):
            sub_results += game.start(networks[i:i + n_agents])
        results.append(sub_results)
    queue.put((worker, results))
    env.close()
