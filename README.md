# genetic-unity
>A framework to train agents in a Unity environment using a genetic algorithm

<p align="center">
  <img src="./animation.gif">
</p>

The project is still in a very early phase, developped and tested only on Windows for the moment

My goal is to provide a framework plugged to Unity environments using ml-agents, that will be used to train neural networks through an evolutionary algorithm

For the moment, only single-agent games can be handled but I plan to make it usable with multi-agents and adversarial games

I use the 3DBall environment (provided with ml-agents) to show examples in this README, if you want to make your own environment, please refer to [this](https://github.com/Unity-Technologies/ml-agents)

This repository contains:
- Ml-agents (the specific version I use since the package is in a very early stage and changes a lot)
- A Genetic Algorithm module and its tool functions
- A Neural Network module
- A Game wrapper that encapsulate the logic of a runing game
- Two example files

The genetic algorithm is parallelized in two different ways which I describe in a following section


## Installation

Python 3.6 was used for this project

Libraries you'll need to run the project:

{``mlagents-envs``, ``numpy``, ``numba``}

Install mlagents-envs:
```sh
cd ml-agents/ml-agents-envs
pip install ./
```

Install numpy:  (I use a specific version of numpy since there is a known problem to pickle certain objects in later versions)
```sh
pip install numpy==1.16.2
```

Install numba:
```sh
pip install numba
```

Clone the repo using

```sh
git clone https://github.com/valentinmace/genetic-unity.git
```

## Usage

You will find some ready to run examples in ``train.py`` and ``run_trained_agent.py`` files.

If you want to train your own neural nets, use:

```sh
python train.py
```

If you want to display a game of my best neural net, use: 
```sh
python run_trained_agent.py
```

## Notes on parallelization

The training phase using the genetic algorithm is paralellized in two ways

#### Pseudo-parallelization:

A very effective way of reducing training time is by adding multipile agents to an environment (instead of juste one)

Taking the example of a mono-process training, there is only one environment in which we can make agents play the game

If we put multiple agents in this environment it helps since they're all playing at the same time, the Game class handles the distribution of multiple neural networks to agents in an environment

<p align="center">
  <img src="./img/pseudo_parallelization.png">
</p>



## Meta

Valentin Macé – [LinkedIn](https://www.linkedin.com/in/valentin-mac%C3%A9-310683165/) – [YouTube](https://www.youtube.com/channel/UCMIW0JKxoxBDM5yiiF17SrA) – [Twitter](https://twitter.com/ValentinMace) - valentin.mace@kedgebs.com

Distributed under the MIT license. See ``LICENSE`` for more information.
