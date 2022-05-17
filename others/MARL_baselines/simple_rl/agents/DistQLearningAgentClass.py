# Python imports.
import random
import numpy as np
from collections import defaultdict

# Other imports.
from simple_rl.agents.AgentClass import Agent

class DistQLearningAgent(Agent):

    def __init__(self, actions, agent_id, name='Distributed Q-learning', gamma=0.99, epsilon=0.3, default_q=0.0):
        Agent.__init__(self, name=name, actions=actions, agent_id=agent_id, gamma=gamma)

        self.epsilon = epsilon
        self.default_q = default_q

        #self.q_func = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : self.default_q))) # limited to 2 agents
        self.q_func = {} # {((s_1, s_2, ...), a_j): q_value}
        self.pi_func = {} # {(s_1, s_2, ...): a_j}

        self.prev_actions = None # available actions

    def get_parameters(self):

        param_dict = defaultdict(float)
        param_dict["gamma"] = self.gamma
        param_dict["epsilon"] = self.epsilon

        return param_dict

    def reset(self):
        self.episode_number = 0
        self.q_func = {}
        self.pi_func = {}
        self.prev_actions = None
        Agent.reset(self)

    def end_of_episode(self):
        self.prev_actions = None
        Agent.end_of_episode(self)

    # main function
    def act(self, state, reward, learning=True, is_final=False, mdp_step=1, continue_sign=False):
        # agent_num = len(state)
        # assert self.agent_id < agent_num
        if not is_final:
            assert not continue_sign
        if learning:
            self.update(self.prev_state, self.prev_action, reward, state, mdp_step, continue_sign)
            action = self.epsilon_greedy_q_policy(state)
        else:
            action = self.greedy_q_policy(state)

        self.prev_state = state
        self.prev_action = action
        self.prev_actions = self.actions # danger

        return action

    def update(self, state, action, reward, next_state, mdp_step, continue_sign):
        if state is None: # first time
            return

        assert action in self.prev_actions
        max_q_s = max([self._check_q_s_a(state, temp_action) for temp_action in self.prev_actions])
        q_s_a = self._check_q_s_a(state, action)

        if not continue_sign:
            max_q_next_s = max([self._check_q_s_a(next_state, temp_action) for temp_action in self.actions])
        else:
            max_q_next_s = self._check_q_s_a(next_state, action) ## very dangerous, please try both ways: with or without using the continue sign
        new_q_s_a = reward + (self.gamma ** mdp_step) * max_q_next_s

        if new_q_s_a > q_s_a:
            self._update_q_s_a(state, action, new_q_s_a)

        new_max_q_s = max([self._check_q_s_a(state, temp_action) for temp_action in self.prev_actions])
        if new_max_q_s != max_q_s:
            assert new_max_q_s > max_q_s
            self._check_pi_s(state)
            self._update_pi_s(state, action)

    def epsilon_greedy_q_policy(self, state):

        if np.random.random() > self.epsilon:
            action = self._check_pi_s(state)
        else:
            action = np.random.choice(self.actions)

        return action

    def greedy_q_policy(self, state):

        return self._check_pi_s(state)

    def _check_q_s_a(self, state, action): # danger
        temp_dict = self.q_func
        agent_num = len(state)
        for agent_id in range(agent_num):
            if state[agent_id] not in temp_dict.keys():
                temp_dict[state[agent_id]] = {}
            temp_dict = temp_dict[state[agent_id]]
        if action not in temp_dict.keys():
            temp_dict[action] = self.default_q

        return temp_dict[action]

    def _check_pi_s(self, state): # danger
        temp_dict = self.pi_func
        agent_num = len(state)
        assert agent_num > 1
        for agent_id in range(agent_num-1):
            if state[agent_id] not in temp_dict.keys():
                temp_dict[state[agent_id]] = {}
            temp_dict = temp_dict[state[agent_id]]
        if state[agent_num-1] not in temp_dict.keys():
            temp_dict[state[agent_num-1]] = np.random.choice(self.actions)

        return temp_dict[state[agent_num-1]]

    def _update_q_s_a(self, state, action, new_q_s_a):
        agent_num = len(state)
        temp_dict = self.q_func
        for agent_id in range(agent_num):
            temp_dict = temp_dict[state[agent_id]]
        temp_dict[action] = new_q_s_a

    def _update_pi_s(self, state, new_pi_s):
        agent_num = len(state)
        temp_dict = self.pi_func
        for agent_id in range(agent_num-1):
            temp_dict = temp_dict[state[agent_id]]
        temp_dict[state[agent_num-1]] = new_pi_s

if __name__ == '__main__':
    pi_func = {}
    actions = ['up', 'down', 'left', 'right']

    def _check_pi_s(state): # danger
        temp_dict = pi_func
        agent_num = len(state)
        assert agent_num > 1
        for agent_id in range(agent_num-1):
            if state[agent_id] not in temp_dict.keys():
                temp_dict[state[agent_id]] = {}
            temp_dict = temp_dict[state[agent_id]]
        if state[agent_num-1] not in temp_dict.keys():
            temp_dict[state[agent_num-1]] = np.random.choice(actions)

        return temp_dict[state[agent_num-1]]

    def _update_pi_s(state, new_pi_s):
        agent_num = len(state)
        temp_dict = pi_func
        for agent_id in range(agent_num-1):
            temp_dict = temp_dict[state[agent_id]]
        temp_dict[state[agent_num-1]] = new_pi_s

    print("1: ", _check_pi_s([0,1]))
    print("2: ", pi_func)

    _update_pi_s([0,1], 'up')
    print("3: ", pi_func)

    _update_pi_s([0, 1], 'down')
    print("4: ", pi_func)

    print("5: ", _check_pi_s([0, 2]))
    print("6: ", pi_func)

    print("7: ", _check_pi_s([1, 2]))
    print("8: ", pi_func)

