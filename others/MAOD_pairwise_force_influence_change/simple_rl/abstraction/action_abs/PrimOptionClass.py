# Python imports.
from collections import defaultdict
import random

# Other imports.
from simple_rl.mdp.StateClass import State


class PrimOption(object):

    def __init__(self, init_predicate, term_predicate, policy, name="primitive_o", term_prob=0.0):
        '''
		Args:
			init_func (S --> {0,1})
			term_func (S --> {0,1})
			policy (S --> A)
		'''
        self.init_predicate = init_predicate
        self.term_predicate = term_predicate
        self.name = name
        self.term_prob = term_prob
        self.policy = policy

    def is_init_true(self, ground_state, agent_id):

        return self.init_predicate.is_true(ground_state[agent_id])

    def is_term_true(self, ground_state, agent_id):

        return self.term_predicate.is_true(ground_state[agent_id])

    def act(self, ground_state, agent_id):

        return self.policy(ground_state[agent_id])

    def __str__(self):

        return "option." + str(self.name)
