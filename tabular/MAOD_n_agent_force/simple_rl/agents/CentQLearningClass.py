# Python imports.
import random
import numpy as np
# import time
from collections import defaultdict

# Other imports.
from simple_rl.agents.AgentClass import Agent

class CentQLearningAgent(Agent):

    def __init__(self, avai_action_list, agent_num, name="Centralized Q-learning", alpha=0.1, gamma=0.99, epsilon=0.3, default_q=0.0):

        Agent.__init__(self, name=name, gamma=gamma)
        self.avai_action_list = avai_action_list

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # the central algorithm is only for 2 agents
        self.agent_num = agent_num

        self.default_q = default_q
        if agent_num == 3:
            self.q_func = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : default_q))))))
        elif agent_num == 4:
            self.q_func = defaultdict(lambda: defaultdict(
                lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: default_q))))))))
        elif agent_num == 5:
            self.q_func = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(
                lambda: defaultdict(lambda: defaultdict(
                    lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: default_q))))))))))
        # {(s_0, s_1, a_0, a_1): q}
        else:
            self.q_func = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: default_q))))
        # {(s_0, s_1, a_0, a_1): q}

    def get_parameters(self):
        param_dict = defaultdict(float)

        param_dict["alpha"] = self.alpha
        param_dict["gamma"] = self.gamma
        param_dict["epsilon"] = self.epsilon

        return param_dict

    def set_avai_action_list(self, avai_action_list):
        # print(len(avai_action_list[0]), " ", len(avai_action_list[1]))
        self.avai_action_list = avai_action_list

    def reset(self):
        self.episode_number = 0
        if self.agent_num == 3:
            self.q_func = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : self.default_q))))))
        elif self.agent_num == 4:
            self.q_func = defaultdict(lambda: defaultdict(
                lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: self.default_q))))))))
        elif self.agent_num == 5:
            self.q_func = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(
                lambda: defaultdict(lambda: defaultdict(
                    lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: self.default_q))))))))))
        # {(s_0, s_1, a_0, a_1): q}
        else:
            self.q_func = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: self.default_q))))
        Agent.reset(self)

    def end_of_episode(self):

        Agent.end_of_episode(self)

    def act(self, state, reward, init_list=[True, True, True, True, True], learning=True, is_final=False, mdp_step=1):

        if not is_final:
            assert (np.array(init_list)).any()

        if learning:
            self.update(self.prev_state, self.prev_action, reward, state, init_list, mdp_step)
            action = self.epsilon_greedy_q_policy(state, init_list, self.prev_action)
        else:
            action = self.greedy_q_policy(state, init_list, self.prev_action)

        assert len(action) == self.agent_num
        self.prev_state = state
        self.prev_action = action

        return action

    def epsilon_greedy_q_policy(self, state, init_list, last_action):

        best_action_list = self.get_max_q_action_list(state, init_list, last_action)
        action_list = []

        for agent_id in range(self.agent_num):
            if np.random.random() > self.epsilon:
                # Exploit.
                action_list.append(best_action_list[agent_id])
            else:
                # Explore
                if init_list[agent_id]:
                    action_list.append(np.random.choice(self.avai_action_list[agent_id]))
                else:
                    # assert best_action_list[agent_id] == last_action[agent_id]
                    action_list.append(best_action_list[agent_id])

        return action_list

    def greedy_q_policy(self, state, init_list, last_action):

        return self.get_max_q_action_list(state, init_list, last_action)

    def update(self, state, action, reward, next_state, init_list, mdp_step):
        if state is None:
            return
        # assert len(state) == len(action) == len(init_list) == self.agent_num, init_list

        max_next_q = self.get_max_q_value(next_state, init_list, action)
        if self.agent_num == 3:
            prev_q_val = self.q_func[state[0]][state[1]][state[2]][action[0]][action[1]][action[2]]

            self.q_func[state[0]][state[1]][state[2]][action[0]][action[1]][action[2]] = (1 - self.alpha) * prev_q_val \
                                                                + self.alpha * (reward + (self.gamma ** mdp_step) * max_next_q)
        elif self.agent_num == 4:
            prev_q_val = self.q_func[state[0]][state[1]][state[2]][state[3]][action[0]][action[1]][action[2]][action[3]]

            self.q_func[state[0]][state[1]][state[2]][state[3]][action[0]][action[1]][action[2]][action[3]] = (1 - self.alpha) * prev_q_val \
                                                                                         + self.alpha * (reward + (
                        self.gamma ** mdp_step) * max_next_q)
        elif self.agent_num == 5:
            prev_q_val = self.q_func[state[0]][state[1]][state[2]][state[3]][state[4]][action[0]][action[1]][action[2]][action[3]][action[4]]

            self.q_func[state[0]][state[1]][state[2]][state[3]][state[4]][action[0]][action[1]][action[2]][action[3]][action[4]] = (1 - self.alpha) * prev_q_val \
                                                                                         + self.alpha * (reward + (
                        self.gamma ** mdp_step) * max_next_q)

    def _compute_max_qval_action_pair(self, state, init_list, last_action):
        assert (self.avai_action_list is not None) and (len(self.avai_action_list) == self.agent_num)
        best_action_list = None


        max_q_val = float("-inf")
        if self.agent_num == 3:
            for action_0 in self.avai_action_list[0]:
                for action_1 in self.avai_action_list[1]:
                    for action_2 in self.avai_action_list[2]:
                        q_val = self.q_func[state[0]][state[1]][state[2]][action_0][action_1][action_2]
                        if q_val > max_q_val:
                            max_q_val = q_val
                            best_action_list = [action_0, action_1, action_2]
        elif self.agent_num == 4:
            for action_0 in self.avai_action_list[0]:
                for action_1 in self.avai_action_list[1]:
                    for action_2 in self.avai_action_list[2]:
                        for action_3 in self.avai_action_list[3]:
                            q_val = self.q_func[state[0]][state[1]][state[2]][state[3]][action_0][action_1][action_2][action_3]
                            if q_val > max_q_val:
                                max_q_val = q_val
                                best_action_list = [action_0, action_1, action_2, action_3]
        elif self.agent_num == 5:
            for action_0 in self.avai_action_list[0]:
                for action_1 in self.avai_action_list[1]:
                    for action_2 in self.avai_action_list[2]:
                        for action_3 in self.avai_action_list[3]:
                            for action_4 in self.avai_action_list[4]:
                                q_val = self.q_func[state[0]][state[1]][state[2]][state[3]][state[4]][action_0][action_1][action_2][action_3][action_4]
                                if q_val > max_q_val:
                                    max_q_val = q_val
                                    best_action_list = [action_0, action_1, action_2, action_3, action_4]


        return max_q_val, best_action_list

    def get_max_q_value(self, state, init_list, last_action):
        return self._compute_max_qval_action_pair(state, init_list, last_action)[0]

    def get_max_q_action_list(self, state, init_list, last_action):
        return self._compute_max_qval_action_pair(state, init_list, last_action)[1]

