# Python imports.
from __future__ import print_function
from collections import defaultdict
# Check python version for queue module.
import sys
if sys.version_info[0] < 3:
    import Queue as queue
else:
    import queue
# Other imports.
from simple_rl.planning.PlannerClass import Planner


class ValueIteration(Planner):

    def __init__(self, mdp, agent_id, name="value_iter", delta=0.0001, max_iterations=500, sample_rate=3):
        '''
        Args:
            mdp (MDP)
            delta (float): After an iteration if VI, if no change more than @\delta has occurred, terminates.
            max_iterations (int): Hard limit for number of iterations.
            sample_rate (int): Determines how many samples from @mdp to take to estimate T(s' | s, a).
            horizon (int): Number of steps before terminating.
        '''
        Planner.__init__(self, mdp, agent_id, name=name)

        self.delta = delta
        self.max_iterations = max_iterations
        self.sample_rate = sample_rate
        self.value_func = defaultdict(float)
        self.reachability_done = False
        self.has_computed_matrix = False
        self.bellman_backups = 0

    def compute_matrix_from_trans_func(self):
        if self.has_computed_matrix:
            self._compute_reachable_state_space()
            # We've already run this, just return.
            return

        self.trans_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float))) # default value = 0.0
        # K: state
        # K: a
        # K: s_prime
        # V: prob

        for s in self.get_states():
            for a in self.actions:
                for sample in range(self.sample_rate):
                    s_prime, _ = self.transition_func(s, a, self.agent_id)
                    self.trans_dict[s][a][s_prime] += 1.0 / self.sample_rate

        self.has_computed_matrix = True

    def get_gamma(self):
        return self.mdp.get_gamma()

    def get_num_states(self):
        if not self.reachability_done:
            self._compute_reachable_state_space()
        return len(self.states)

    def get_states(self):
        if self.reachability_done:
            return list(self.states)
        else:
            self._compute_reachable_state_space()
            return list(self.states)

    def get_value(self, s):
        '''
        Args:
            s (State)

        Returns:
            (float)
        '''
        return self._compute_max_qval_action_pair(s)[0]

    def get_q_value(self, s, a):
        '''
        Args:
            s (State)
            a (str): action

        Returns:
            (float): The Q estimate given the current value function @self.value_func.
        '''
        # Compute expected value.
        expected_future_val = 0
        for s_prime in self.trans_dict[s][a].keys(): # s'
            expected_future_val += self.trans_dict[s][a][s_prime] * self.value_func[s_prime]

        return self.reward_func(s, a, self.agent_id) + self.gamma * expected_future_val

    def _compute_reachable_state_space(self):
        '''
        Summary:
            Starting with @self.start_state, determines all reachable states
            and stores them in self.states.
        '''

        if self.reachability_done:
            return

        state_queue = queue.Queue()
        state_queue.put(self.init_state)
        self.states.add(self.init_state) # set, so GridWorldState must be hashable!!!

        while not state_queue.empty():
            s = state_queue.get()
            for a in self.actions:
                for samples in range(self.sample_rate):  # Take @sample_rate samples to estimate E[V]
                    next_state, _ = self.transition_func(s, a, self.agent_id)

                    if next_state not in self.states:
                        self.states.add(next_state) # automatic rank based on the hash value of the elements
                        state_queue.put(next_state)

        self.reachability_done = True
        assert type(self.states) is set
        # print("The size of the state space is ", str(len(self.states)))

    # main function
    def run_vi(self):
        '''
        Returns:
            (tuple):
                1. (int): num iterations taken.
                2. (float): value.
        Summary:
            Runs ValueIteration and fills in the self.value_func.           
        '''
        # Algorithm bookkeeping params.
        iterations = 0
        max_diff = float("inf")
        self.compute_matrix_from_trans_func()
        state_space = self.get_states()
        self.bellman_backups = 0

        # Main loop.
        while max_diff > self.delta and iterations < self.max_iterations:
            max_diff = 0
            for s in state_space:
                # print(self.value_func[s])
                self.bellman_backups += 1
                if self.mdp.is_goal_state_single(s, self.agent_id):
                    continue

                max_q = float("-inf")
                for a in self.actions:
                    q_s_a = self.get_q_value(s, a)
                    max_q = q_s_a if q_s_a > max_q else max_q

                # Check terminating condition.
                max_diff = max(abs(self.value_func[s] - max_q), max_diff)

                # Update value.
                self.value_func[s] = max_q
            iterations += 1

        value_of_init_state = self._compute_max_qval_action_pair(self.init_state)[0]
        self.has_planned = True
        # print("After the {} value iterations, the error is {}!".format(iterations, max_diff))
        return iterations, value_of_init_state

    def get_num_backups_in_recent_run(self):
        if self.has_planned:
            return self.bellman_backups
        else:
            print("Warning: asking for num Bellman backups, but VI has not been run.")
            return 0

    def print_value_func(self):
        for key in self.value_func.keys():
            print(key, ":", self.value_func[key])

    def _get_max_q_action(self, state):
        '''
        Args:
            state (State)

        Returns:
            (str): The action with the max q value in the given @state.
        '''
        return self._compute_max_qval_action_pair(state)[1]

    def policy(self, state):
        '''
        Args:
            state (State)

        Returns:
            (str): Action

        Summary:
            For use in a FixedPolicyAgent.
        '''
        return self._get_max_q_action(state)

    def _compute_max_qval_action_pair(self, state):
        '''
        Args:
            state (State)

        Returns:
            (tuple) --> (float, str): where the float is the Qval, str is the action.
        '''
        # Grab random initial action in case all equal
        max_q_val = float("-inf")
        best_action = self.actions[0]

        # Find best action (action w/ current max predicted Q value)
        for action in self.actions:
            q_s_a = self.get_q_value(state, action)
            if q_s_a > max_q_val:
                max_q_val = q_s_a
                best_action = action

        return max_q_val, best_action

    def compute_adjacency_matrix(self):
        import numpy as np
        assert self.reachability_done and self.has_computed_matrix

        closedList = []
        states = list(self.states)
        sToInd = {v: k for k, v in enumerate(states)} # the __eq__ method be defined in GridWorldState
        N = len(self.states)
        A = np.zeros((N, N), dtype=int)

        state_queue = queue.Queue()
        state_queue.put(self.init_state)
        closedList.append(self.init_state)

        while not state_queue.empty():
            s = state_queue.get()
            for a in self.actions:
                for samples in range(self.sample_rate):  # Take @sample_rate samples to estimate E[V]
                    next_state, _ = self.transition_func(s, a, self.agent_id)
                    A[sToInd[s]][sToInd[next_state]] = 1
                    A[sToInd[next_state]][sToInd[s]] = 1  # TODO: Here we have a undirected graph
                    if next_state not in closedList:
                        closedList.append(next_state)
                        state_queue.put(next_state)
        return A, states