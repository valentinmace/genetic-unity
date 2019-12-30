# Valentin Macé
# valentin.mace@kedgebs.com
# Developed for fun
# Feel free to use this code as you wish as long as you quote me as author

"""
game.py
~~~~~~~~~~

A module to implement a game taking place in a UnityEnvironment

"""

import numpy as np
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from neural_network import *



class Game:
    """ Game Class

    Game is a wrapper that takes a UnityEnvironment and makes a set of NeuralNetworks play in this environment.
    The game stops when all agents are done and returns their scores. An Agent does not play again when he's done.
    A Game might contain multiple simulations to make multiple neural networks play at the same time.

    """

    def __init__(self, unity_env, time_scale=1.0, width=720, height=480, target_frame_rate=60, quality_level=5):
        """ Initializes the game

        :param unity_env: (UnityEnvironment) Environment where the game will be played
        :param time_scale:(float) Speed of the game
        :param width:(int) Window's width
        :param height:(int) Window's height
        :param target_frame_rate:(int) Frame rate
        :param quality_level:(int) Visual quality

        Todo: Commentate a little, reorganise
        """
        self.unity_env = unity_env
        self.unity_env.reset()
        engine_configuration_channel = EngineConfigurationChannel()
        engine_configuration_channel.set_configuration_parameters(time_scale=time_scale, width=width, height=height,
                                                                  target_frame_rate=target_frame_rate, quality_level=quality_level)
        self.unity_env.side_channels[2] = engine_configuration_channel

        self.group_name = unity_env.get_agent_groups()[0]
        self.group_spec = unity_env.get_agent_group_spec(self.group_name)
        self.n_agents = self.unity_env.get_step_result(self.group_name).n_agents()
        self.action_size = self.group_spec.action_size

    def start(self, neural_nets):
        """
        Play the game until all agents are done

        :param neural_nets:(list of NeuralNetwork) Neural nets that will play the game
        :return:(list of float) Scores for each NeuralNetwork

        Todo: Make code more readable, factorise and commentate
        """
        n_nets = len(neural_nets)
        self.unity_env.reset()
        step_result = self.unity_env.get_step_result(self.group_name)
        done = [False if i < n_nets else True for i in range(self.n_agents)]
        score = [0 for _ in range(n_nets)]
        while not all(done):
            observations = [step_result.obs[0][i].reshape((-1,1)) for i in range(self.n_agents)]
            actions = np.array([np.array((neural_nets[i].feed_forward(observations[i]).flatten()-0.5)*2) if not done[i] else np.array([0, 0])
                                for i in range(self.n_agents)])
            self.unity_env.set_actions(self.group_name, actions)
            self.unity_env.step()
            step_result = self.unity_env.get_step_result(self.group_name)
            mask = step_result.done
            score = [score[i]+step_result.reward[i] if not done[i] else score[i] for i in range(n_nets)]
            done = [a or b for a, b in zip(mask, done)]
        return score
