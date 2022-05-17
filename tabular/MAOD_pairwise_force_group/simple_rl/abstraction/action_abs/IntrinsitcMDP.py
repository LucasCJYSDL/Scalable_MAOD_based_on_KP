import gym
from simple_rl.mdp.MDPClass import MDP
from collections import defaultdict
import copy

class IntrinsicMDP(MDP):
    '''
    GymMDP with intrinsitc reward added
    '''
    def __init__(self, intrinsic_term_list, mdp=None):
        '''
        Intrinsic reward should be a function (GymState -> float).
        '''
        self.intrinsic_term_list = intrinsic_term_list
        self.intrinsic_term_locs = []
        for term in self.intrinsic_term_list:
            self.intrinsic_term_locs.append((term.x, term.y))

        # if mdp is not None:
        assert mdp is not None
        self.env = copy.deepcopy(mdp)
        MDP.__init__(self, mdp.actions, None, None, self._transition_func, self._reward_func, init_states=mdp.get_init_states())


    def _reward_func(self, state, action, agent_id):
        '''
        Args:
            state (AtariState)
            action (str)
        Returns
            (float)
        '''
        # reward = self.env._reward_func_single(state, action)
        reward = 0.0

        if self._is_intrinsic_goal_state_action(state, action):
            return reward + 1.0 - self.env.step_cost
        else:
            return reward + 0.0 - self.env.step_cost


    def _is_intrinsic_goal_state_action(self, state, action):
        if isinstance(action, str):
            action = self.env.action_conv(action)

        if (state.x, state.y) in self.intrinsic_term_locs:
            return False

        if action == 3 and (state.x - 1, state.y) in self.intrinsic_term_locs:
            return True
        elif action == 2 and (state.x + 1, state.y) in self.intrinsic_term_locs:
            return True
        elif action == 1 and (state.x, state.y - 1) in self.intrinsic_term_locs:
            return True
        elif action == 0 and (state.x, state.y + 1) in self.intrinsic_term_locs:
            return True
        else:
            return False


    def _transition_func(self, state, action, agent_id):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (State)
        '''
        next_state, is_terminal = self.env._transition_func_single(state, action, agent_id = agent_id, allow_further=True)
        # if allow_further == False, the goal states will not able to reach the intrinsic goal
        return next_state, is_terminal

    def is_goal_state_single(self, state, agent_id):
        return (int(state.x), int(state.y)) in self.intrinsic_term_locs

    def reset(self):
        self.env.reset()

