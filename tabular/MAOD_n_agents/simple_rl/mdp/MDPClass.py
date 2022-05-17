''' MDPClass.py: Contains the MDP Class. '''

# Python imports.
import copy

class MDP(object):
    ''' Abstract class for a Markov Decision Process. '''
    
    def __init__(self, actions, transition_func, reward_func, transition_func_single, \
                 reward_func_single, init_states, gamma=0.99, step_cost=0): #c
        self.actions = actions
        self.transition_func = transition_func
        self.reward_func = reward_func
        self.transition_func_single = transition_func_single
        self.reward_func_single = reward_func_single
        self.gamma = gamma
        self.init_states = copy.deepcopy(init_states) # no interruption
        self.cur_states = init_states
        self.step_cost = step_cost

    # ---------------
    # -- Accessors --
    # ---------------

    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        param_dict = {}
        param_dict["gamma"] = self.gamma
        param_dict["step_cost"] = self.step_cost

        return param_dict

    def get_init_states(self): #c
        return self.init_states

    def get_curr_states(self): #c
        return self.cur_states

    def get_actions(self):
        return self.actions

    def get_gamma(self):
        return self.gamma

    def get_reward_func_single(self):
        return self.reward_func_single

    def get_transition_func_single(self):
        return self.transition_func_single

    # --------------
    # -- Mutators --
    # --------------

    def set_gamma(self, new_gamma):
        self.gamma = new_gamma

    def set_step_cost(self, new_step_cost):
        self.step_cost = new_step_cost

    # ----------
    # -- Core --
    # ----------

    def execute_agent_action(self, action_list): #c
        '''
        Summary:
            Core method of all of simple_rl. Facilitates interaction
            between the MDP and an agent.
        '''
        rewards = self.reward_func(self.cur_states, action_list)
        next_states, is_terminal = self.transition_func(self.cur_states, action_list)
        self.cur_states = next_states

        return rewards, next_states, is_terminal # float, List of GridWorldState, Bool

    def execute_agent_action_single(self, action, agent_id):
        '''
        Args:
            action (str)

        Returns:
            (tuple: <float,State>): reward, State

        Summary:
            Core method of all of simple_rl. Facilitates interaction
            between the MDP and an agent.
        '''
        reward = self.reward_func_single(self.cur_states[agent_id], action)
        next_state, is_terminal = self.transition_func_single(self.cur_states[agent_id], action)
        self.cur_states[agent_id] = next_state

        return reward, next_state, is_terminal

    def reset(self): #c
        self.cur_states = copy.deepcopy(self.init_states)

    def end_of_instance(self):
        print("This instance is over!")
