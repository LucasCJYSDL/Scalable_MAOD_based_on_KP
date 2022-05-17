# Python imports.
from __future__ import print_function
from collections import defaultdict
import random
import numpy as np

# Other imports.
from simple_rl.abstraction.action_abs.OptionClass import Option
from simple_rl.abstraction.action_abs.PrimOptionClass import PrimOption
from simple_rl.abstraction.action_abs.PredicateClass import Predicate
# from simple_rl.agents import CentQLearningAgent, MultiCentQLearningAgent

class ActionAbstraction(object):

    def __init__(self, options, prim_actions, group_id, agent_num, term_prob=0.0, prims_on_failure=False, use_prims=True, gamma=0.99, name=None): # prim: primitive

        self.group_id = group_id
        self.agent_num = agent_num
        assert self.agent_num == 2
        if 'single' in name:
            self.option_type = 'single'
        else:
            assert 'multiple' in name
            self.option_type = 'multiple'

        self.options = []
        if use_prims:
            if options is not None:
                if self.option_type == 'single':
                    for agent_id in range(self.group_id*self.agent_num, self.group_id*self.agent_num+self.agent_num):
                        self.options.append(self._convert_to_options(prim_actions) + options[agent_id])
                else:
                    for agent_id in range(self.agent_num):
                        self.options.append(self._convert_to_options(prim_actions) + options[self.group_id])
            else:
                for agent_id in range(self.agent_num):
                    self.options.append(self._convert_to_options(prim_actions))
        else:
            assert False

        self.is_cur_executing = False
        self.cur_option = [None for _ in range(self.agent_num)] # The option we're executing currently.
        self.term_prob = term_prob
        self.prims_on_failure = prims_on_failure
        if self.prims_on_failure:
            self.prim_actions = prim_actions
        self.high_level_reward = 0.0
        self.intra_option_step = 0
        self.gamma = gamma

    def act(self, agent, abstr_state, ground_state, reward, is_final=False, learning=True):
        '''
        Args:
            agent (Agent)
            abstr_state (State)
            ground_state (State)
            reward (float)
        Returns:
            (str)
        '''
        ################################
        # NOTE: Current Implementation always select options over actions.
        #       Give primitive actions to the agent by use_prims=True
        ################################

        # print('ActionAbstractionClass: type(ground_state)=', type(ground_state))
        self.high_level_reward += (self.gamma ** self.intra_option_step) * reward

        assert type(ground_state) is list
        # if self.option_type == 'multiple':
        #     assert isinstance(agent, MultiCentQLearningAgent)
        # else:
        #     assert isinstance(agent, CentQLearningAgent)

        continue_list = self.is_next_step_continuing_option(ground_state)

        if self.option_type == 'single':
            continue_sign = (np.array(continue_list)).all()
        else:
            continue_sign = (np.array(continue_list)).any()

        if continue_sign and random.random() > self.term_prob and (not is_final):
            # the high level policy must be updated, no matter whether the last option terminates in the middle or not
            # We're in an option and not terminating.
            self.intra_option_step += 1
            return self.get_next_ground_action(ground_state, continue_list)
        else:
            # We're not in an option, check with agent.
            active_options = self.get_active_options(ground_state)

            assert len(active_options) == self.agent_num
            agent.set_avai_action_list(active_options)

            if self.option_type == 'single':
                init_list = []
                assert len(continue_list) == self.agent_num
                for agent_id in range(self.agent_num):
                    init_list.append(not continue_list[agent_id])
            else:
                flag = True
                for agent_id in range(self.agent_num):
                    if continue_list[agent_id]:
                        flag=False
                        break
                init_list = [flag for _ in range(self.agent_num)]

            option_list = agent.act(abstr_state, self.high_level_reward, init_list=init_list, is_final=is_final, mdp_step=self.intra_option_step+1, learning=learning) # the high-level policy is updated in 'act'
            self.high_level_reward = 0.0
            self.intra_option_step = 0
            self.set_option_executing(option_list)

            return self.abs_to_ground(ground_state, option_list)

    def get_active_options(self, state):
        '''
        Args:
            state (State)

        Returns:
            (list): Contains all active options.
        '''
        assert type(state) is list

        active_options = []
        for agent_id in range(self.agent_num):
            active_options.append([o for o in self.options[agent_id] if o.is_init_true(state, agent_id=(self.group_id * self.agent_num + agent_id))])

        return active_options

    def _convert_to_options(self, action_list):
        '''
        Args:
            action_list (list)
        Returns:
            (list of Option)
        '''
        options = []
        for ground_action in action_list:
            o = PrimOption(init_predicate=Predicate(make_lambda(True)),
                        term_predicate=Predicate(make_lambda(True)),
                        policy=make_lambda(ground_action),
                        name="prim." + str(ground_action))
            options.append(o)

        return options

    def is_next_step_continuing_option(self, ground_state):

        assert type(ground_state) is list

        continue_list = []
        for agent_id in range(self.agent_num):
            continue_list.append(self.is_cur_executing and not self.cur_option[agent_id].is_term_true(ground_state, agent_id=(self.group_id * self.agent_num + agent_id)))

        return continue_list

    def set_option_executing(self, option_list):

        self.cur_option = option_list
        self.is_cur_executing = True

    def get_next_ground_action(self, ground_state, continue_list):
        assert type(ground_state) is list
        action_list = []
        for agent_id in range(self.agent_num):
            if continue_list[agent_id]:
                action_list.append(self.cur_option[agent_id].act(ground_state, agent_id=(self.group_id * self.agent_num + agent_id)))
            else:
                action_list.append('stay')

        return action_list

    def get_actions(self):
        return list(self.options)

    def abs_to_ground(self, ground_state, option_list):
        assert type(ground_state) is list
        action_list = []
        for agent_id in range(self.agent_num):
            action_list.append(option_list[agent_id].act(ground_state, agent_id=(self.group_id * self.agent_num + agent_id)))

        return action_list

    # def add_option(self, option):
    #     self.options += [option]

    def reset(self):
        self.high_level_reward = 0.0
        self.intra_option_step = 0
        self.is_cur_executing = False
        self.cur_option = [None for _ in range(self.agent_num)] # The option we're executing currently.

    def end_of_episode(self):
        self.reset()


def make_lambda(result):
    return lambda x : result
