'''
Implementations of standard RL agents:

        AgentClass: Contains the basic skeleton of an RL Agent.
        QLearningAgentClass: Q-Learning.
        LinearQAgentClass: Q-Learning with a Linear Approximator.
        RandomAgentClass: Random actor.
        RMaxAgentClass: R-Max.
        LinUCBAgentClass: Contextual Bandit Algorithm.
'''

# Grab agent classes.
from simple_rl.agents.AgentClass import Agent
from simple_rl.agents.QLearningAgentClass import QLearningAgent
from simple_rl.agents.RandomAgentClass import RandomAgent
from simple_rl.agents.DistQLearningAgentClass import DistQLearningAgent


try:
        from simple_rl.agents.func_approx.DQNAgentClass import DQNAgent
        from simple_rl.agents.func_approx.DDPGAgentClass import DDPGAgent
except ImportError:
        print("Warning: Tensorflow not installed.")
        pass

