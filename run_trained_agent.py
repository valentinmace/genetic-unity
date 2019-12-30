# Valentin Macé
# valentin.mace@kedgebs.com
# Developed for fun
# Feel free to use this code as you wish as long as you quote me as author

"""
run_trained_agent.py
~~~~~~~~~~

Example code showing how to run a trained NeuralNetwork in a UnityEnvironment and display it

"""

from mlagents_envs.environment import UnityEnvironment

from game import *
from neural_network import *

if __name__ == '__main__':

    """
    Paths to Unity games

    I've built two environments for this example:
    The first has only one agent and the second has multiple agents
    """
    single_agent_env_name = "./examples/builds/3DBall/single_agent/ball.exe"
    multiple_agents_env_name = "./examples/builds/3DBall/multiple_agents/ball.exe"


    """
    Display a game of 3DBall played by my best neural network in a single agent 3DBall environment

    It is very good at stabilizing the ball so there is
    not much to see
    """
    env = UnityEnvironment(base_port=5006, file_name=single_agent_env_name, seed=5)                 # Instantiate environment
    env.reset()                                                                                     # Reset environment
    net = NeuralNetwork([8, 16, 2])
    net.load(filename_weights='saved/goat_weights.npy', filename_biases='saved/goat_biases.npy')    # Load my best neural net
    game = Game(env, time_scale=2.0)
    game.start(neural_nets=[net])                                                                   # Play game


    """
    Display a game of 3DBall played by my best neural network in a multi-agents 3DBall environment

    A multi-agents environment is very useful for training, for example here 12 agents play in parallel
    You might need to commentate previous single agent example since it will take a while before it reaches max_steps
    """
    env = UnityEnvironment(base_port=5006, file_name=multiple_agents_env_name, seed=1)
    env.reset()
    nets = [NeuralNetwork([8, 16, 2]) for i in range(12)]
    for net in nets: net.load(filename_weights='saved/goat_weights.npy', filename_biases='saved/goat_biases.npy')
    game = Game(env, time_scale=2.0)
    game.start(neural_nets=nets)