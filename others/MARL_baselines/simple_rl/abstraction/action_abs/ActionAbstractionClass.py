# Python imports.
from __future__ import print_function
from collections import defaultdict
import random
import numpy as np

# Other imports.
from simple_rl.abstraction.action_abs.OptionClass import Option
from simple_rl.abstraction.action_abs.PrimOptionClass import PrimOption
from simple_rl.abstraction.action_abs.PredicateClass import Predicate

class ActionAbstraction(object):

    def __init__(self, options, prim_actions, term_prob=0.0, prims_on_failure=False, use_prims=True, agent_id=None, gamma=0.99, name=None): # prim: primitive
        self.agent_id = agent_id
        if 'single' in name:
            self.option_type = 'single'
        else:
            assert 'multiple' in name
            self.option_type = 'multiple'

        if use_prims:
            if options is not None:
                if self.option_type == 'single':
                    self.options = self._convert_to_options(prim_actions) + options[agent_id]
                else:
                    self.options = self._convert_to_options(prim_actions) + options
                print("Total number of the options for the {} agent is {}!".format(self.option_type, len(self.options)))
            else:
                self.options = self._convert_to_options(prim_actions)
        else:
            assert options is not None
            if self.option_type == 'single':
                self.options = options[agent_id]
            else:
                self.options = options

        self.is_cur_executing = False
        self.cur_option = None # The option we're executing currently.
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

        continue_sign = self.is_next_step_continuing_option(ground_state)

        if continue_sign and random.random() > self.term_prob and (not is_final):
            # the high level policy must be updated, no matter whether the last option terminates in the middle or not
            # We're in an option and not terminating.
            self.intra_option_step += 1
            return self.get_next_ground_action(ground_state)
        else:
            # We're not in an option, check with agent.
            active_options = self.get_active_options(ground_state)

            if len(active_options) == 0:
                if self.prims_on_failure:
                    # In a failure state, back off to primitives.
                    agent.actions = self._convert_to_options(self.prim_actions)
                else:
                    # No actions available.
                    raise ValueError("(simple_rl) Error: no actions available in state " + str(ground_state) + ".")
            else:
                # Give agent available options.
                agent.actions = active_options
            
            abstr_action = agent.act(abstr_state, self.high_level_reward, is_final=is_final, mdp_step=self.intra_option_step+1, continue_sign=continue_sign, learning=learning) # the high-level policy is updated in 'act'
            self.high_level_reward = 0.0
            self.intra_option_step = 0
            self.set_option_executing(abstr_action)

            return self.abs_to_ground(ground_state, abstr_action)

    def get_active_options(self, state):
        '''
        Args:
            state (State)

        Returns:
            (list): Contains all active options.
        '''
        assert type(state) is list
        return [o for o in self.options if o.is_init_true(state, self.agent_id)]

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
        '''
        Returns:
            (bool): True iff an option was executing and should continue next step.
        '''
        assert type(ground_state) is list
        #     return self.is_cur_executing and not (np.array(self.cur_option.is_term_true(ground_state)).all())
        # else:
        return self.is_cur_executing and not self.cur_option.is_term_true(ground_state, agent_id=self.agent_id)

    def set_option_executing(self, option):
        if option not in self.options and "prim" not in option.name:
            raise ValueError("(simple_rl) Error: agent chose a non-existent option (" + str(option) + ").")

        self.cur_option = option
        self.is_cur_executing = True

    def get_next_ground_action(self, ground_state):
        assert type(ground_state) is list
        return self.cur_option.act(ground_state, agent_id=self.agent_id)

    def get_actions(self):
        return list(self.options)

    def abs_to_ground(self, ground_state, abstr_action):
        assert type(ground_state) is list
        return abstr_action.act(ground_state, agent_id=self.agent_id) # with or without agent_id ???

    # def add_option(self, option):
    #     self.options += [option]

    def reset(self):
        self.high_level_reward = 0.0
        self.intra_option_step = 0
        self.is_cur_executing = False
        self.cur_option = None # The option we're executing currently.

    def end_of_episode(self):
        self.reset()


def make_lambda(result):
    return lambda x : result
