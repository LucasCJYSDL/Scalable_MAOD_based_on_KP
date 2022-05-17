# Python imports.
from collections import defaultdict
import random

# Other imports.
from simple_rl.mdp.StateClass import State


class Option(object):

    def __init__(self, init_predicate, term_predicate, policy, name="o", term_prob=0.0, is_single=True, group_id=-1):
        '''
		Args:
			init_func (S --> {0,1})
			term_func (S --> {0,1})
			policy (S --> A)
		'''
        self.init_predicate = init_predicate
        self.term_predicate = term_predicate
        # self.term_flag = False
        self.name = name
        self.term_prob = term_prob
        self.policy = policy

        self.is_single = is_single
        if not is_single:
            assert len(term_predicate) == len(policy) == 2
            self.agent_num = len(term_predicate)
            self.group_id = group_id
            # self.flag_list = None # whether agent i has chosen the multi-agent option

    # def update_flag_list(self, agent_list): # this myst be recalled when a new multi-agent option is chosen
    #     self.flag_list = [False for _ in range(self.agent_num)]
    #     for agent_id in agent_list:
    #         self.flag_list[agent_id] = True

    def is_init_true(self, ground_state, agent_id):
        if self.is_single:
            return self.init_predicate.is_true(ground_state[agent_id]) and (not self.is_term_true(ground_state, agent_id))
        else:
            assert agent_id >= self.group_id * self.agent_num and agent_id < (self.group_id + 1) * self.agent_num
            state_list = []
            for idx in range(self.agent_num):
                state_idx = self.group_id * self.agent_num + idx
                state_list.append(ground_state[state_idx])
                # if not self.init_predicate[idx].is_true(ground_state[state_idx]):
                #     return False
            ## if not self.is_term_true(ground_state, agent_id):
            #    # return True
            ## else:
            #     return False
            # return True
            try:
                return self.init_predicate[tuple(state_list)]
            except:
                return False

    def is_term_true(self, ground_state, agent_id):
        if self.is_single:
            # return self.term_predicate.is_true(ground_state) or self.term_prob > random.random()
            return self.term_predicate.is_true(ground_state[agent_id])
        else:
            # term_list = []
            # for agent_id in range(self.agent_num):
            #     term_list.append(self.term_predicate[agent_id].is_true(ground_state[agent_id])
            #                      or (not self.flag_list[agent_id]) or (self.term_prob > random.random()))
            # return term_list
            assert agent_id >= self.group_id * self.agent_num and agent_id < (self.group_id + 1) * self.agent_num
            term_idx = agent_id % self.agent_num
            return self.term_predicate[term_idx].is_true(ground_state[agent_id]) # discentralized, since the mmulti_agent option choice is not forced!

    def act(self, ground_state, agent_id):
        if self.is_single:
            return self.policy(ground_state[agent_id])
        else:
            # assert self.flag_list[agent_id], "This agent is not using this option!"
            assert agent_id >= self.group_id * self.agent_num and agent_id < (self.group_id + 1) * self.agent_num
            policy_idx = agent_id % self.agent_num
            return self.policy[policy_idx](ground_state[agent_id])

    def set_policy(self, policy):
        assert self.is_single
        self.policy = policy

    def set_name(self, new_name):
        self.name = new_name

    def __str__(self):
        return "option." + str(self.name)
