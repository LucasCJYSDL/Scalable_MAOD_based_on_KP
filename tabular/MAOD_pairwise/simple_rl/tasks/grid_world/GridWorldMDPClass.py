''' GridWorldMDPClass.py: Contains the GridWorldMDP class. '''

# Python imports.
from __future__ import print_function
import random
import os
import numpy as np
from collections import defaultdict

# Other imports.
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState
import copy

class GridWorldMDP(MDP):
    ''' Class for a Grid World MDP '''

    # Static constants.
    ACTIONS = ["up", "down", "left", "right"]

    def __init__(self, width, height, init_locs, goal_locs, lava_locs, walls,
                rand_init=False, is_goal_terminal=True,
                gamma=0.99, slip_prob=0.0, step_cost=0.0, lava_cost=0.01, name="gridworld"): #c
        '''
        Args:
            height (int)
            width (int)
            init_locs (list of tuples: [(int, int)...])
            goal_locs (list of tuples: [(int, int)...])
            lava_locs (list of tuples: [(int, int)...]): These locations return -1 reward.
            walls (list)
            is_goal_terminal (bool)
        '''
        # Setup init location.
        self.rand_init = rand_init
        self.agent_num = len(init_locs)
        if rand_init:
            init_locs = []
            for _ in range(self.agent_num):
                init_loc = random.randint(1, width), random.randint(1, height)
                while init_loc in walls:
                    init_loc = random.randint(1, width), random.randint(1, height)
                init_locs.append(init_loc)

        self.init_locs = init_locs
        init_states = []
        for i in range(self.agent_num):
            init_states.append(GridWorldState(init_locs[i][0], init_locs[i][1]))

        MDP.__init__(self, GridWorldMDP.ACTIONS, self._transition_func, self._reward_func, self._transition_func_single, \
                     self._reward_func_single, init_states=init_states, gamma=gamma)

        if type(goal_locs) is not list:
            raise ValueError("(simple_rl) GridWorld Error: argument @goal_locs needs to be a list of locations. For example: [(3,3), (4,3)].")
        self.step_cost = step_cost
        self.lava_cost = lava_cost
        self.walls = walls
        self.width = width
        self.height = height
        self.goal_locs = goal_locs
        self.cur_states = []
        for i in range(self.agent_num):
            self.cur_states.append(GridWorldState(init_locs[i][0], init_locs[i][1]))

        self.is_goal_terminal = is_goal_terminal
        self.slip_prob = slip_prob
        self.name = name
        self.lava_locs = lava_locs

    def reset(self): #c
        if self.rand_init:
            self.cur_states = []
            for _ in range(self.agent_num):
                init_loc = random.randint(1, self.width), random.randint(1, self.height)
                while init_loc in self.walls:
                    init_loc = random.randint(1, self.width), random.randint(1, self.height)
                self.cur_states.append(GridWorldState(init_loc[0], init_loc[1]))
        else:
            self.cur_states = copy.deepcopy(self.init_states)

    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        param_dict = defaultdict(int)
        param_dict["width"] = self.width
        param_dict["height"] = self.height
        param_dict["init_locs"] = self.init_locs
        param_dict["rand_init"] = self.rand_init
        param_dict["goal_locs"] = self.goal_locs
        param_dict["lava_locs"] = self.lava_locs
        param_dict["walls"] = self.walls
        param_dict["is_goal_terminal"] = self.is_goal_terminal
        param_dict["gamma"] = self.gamma
        param_dict["slip_prob"] = self.slip_prob
        param_dict["step_cost"] = self.step_cost
        param_dict["lava_cost"] = self.lava_cost
   
        return param_dict

    def set_slip_prob(self, slip_prob):
        self.slip_prob = slip_prob

    def get_slip_prob(self):
        return self.slip_prob

    # multi-agent version
    def is_goal_state(self, states): #c #check
        assert len(states) == self.agent_num, len(states)
        for i in range(self.agent_num):
            if (int(states[i].x), int(states[i].y)) not in self.goal_locs:
                return False
        return True and self.is_goal_terminal

    def _reward_func(self, states, action_list): #c
        '''
        Args:
            state (State)
            action (str)

        Returns
            (float)
        '''
        if self._is_goal_state_action(states, action_list):
            assert 1.0 - self.step_cost >= 0.0 # assumption for the distributed q-learning
            return 1.0 - self.step_cost
        else:
            assert 0.0 - self.step_cost >= 0.0
            return 0.0 - self.step_cost

    def _is_goal_state_action(self, states, action_list): #c
        '''
        Returns:
            (bool): True iff the state-action pair send the agent to the goal state.
        '''
        assert (len(states) == self.agent_num) and (len(action_list) == self.agent_num)

        if self.is_goal_state(states):
            return False

        next_states, is_terminal = self._transition_func(states, action_list)
        if is_terminal:
            return True
        else:
            return False

    def _transition_func(self, states, action_list): #c
        '''
        Returns
            (State)
        '''
        assert (len(states) == self.agent_num) and (len(action_list) == self.agent_num)

        if self.is_goal_state(states):
            return states, True

        next_states = []
        for i in range(self.agent_num):
            action = action_list[i]
            state = states[i]

            if isinstance(action, str):
               action = self.action_conv(action)
            r = random.random()
            if self.slip_prob > r:
                # Flip dir.
                if action == 0:
                    action = random.choice([2, 3])
                elif action == 1:
                    action = random.choice([2, 3])
                elif action == 2:
                    action = random.choice([0, 1])
                elif action == 3:
                    action = random.choice([0, 1])

            if action == 0 and state.y < self.height and not self.is_wall(state.x, state.y + 1):
                next_state = GridWorldState(state.x, state.y + 1)
            elif action == 1 and state.y > 1 and not self.is_wall(state.x, state.y - 1):
                next_state = GridWorldState(state.x, state.y - 1)
            elif action == 2 and state.x < self.width and not self.is_wall(state.x + 1, state.y):
                next_state = GridWorldState(state.x + 1, state.y)
            elif action == 3 and state.x > 1 and not self.is_wall(state.x - 1, state.y):
                next_state = GridWorldState(state.x - 1, state.y)
            else:
                next_state = GridWorldState(state.x, state.y)

            next_states.append(next_state)

        if self.is_goal_state(next_states):
            is_terminal = True
        else:
            is_terminal = False

        return next_states, is_terminal

    #single-agent
    def is_goal_state_single(self, state):
        return ((int(state.x), int(state.y)) in self.goal_locs) and self.is_goal_terminal

    def _reward_func_single(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (float)
        '''
        if self._is_goal_state_action_single(state, action):
            return 1.0 - self.step_cost
        else:
            return 0 - self.step_cost

    def _is_goal_state_action_single(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns:
            (bool): True iff the state-action pair send the agent to the goal state.
        '''
        if isinstance(action, str):
            action = self.action_conv(action)

        if (state.x, state.y) in self.goal_locs:
            # Already at terminal.
            return False

        if action == 3 and (state.x - 1, state.y) in self.goal_locs:
            return True
        elif action == 2 and (state.x + 1, state.y) in self.goal_locs:
            return True
        elif action == 1 and (state.x, state.y - 1) in self.goal_locs:
            return True
        elif action == 0 and (state.x, state.y + 1) in self.goal_locs:
            return True
        else:
            return False

    def _transition_func_single(self, state, action, allow_further=False):
        '''
        Args:
            state (State)
            action (str)
        Returns
            (State)
        '''
        if isinstance(action, str):
            action = self.action_conv(action)

        if not allow_further:
            if self.is_goal_state_single(state):
                return state, True

        r = random.random()
        if self.slip_prob > r:
            # Flip dir.
            if action == 0:
                action = random.choice([2, 3])
            elif action == 1:
                action = random.choice([2, 3])
            elif action == 2:
                action = random.choice([0, 1])
            elif action == 3:
                action = random.choice([0, 1])

        if action == 0 and state.y < self.height and not self.is_wall(state.x, state.y + 1):
            next_state = GridWorldState(state.x, state.y + 1)
        elif action == 1 and state.y > 1 and not self.is_wall(state.x, state.y - 1):
            next_state = GridWorldState(state.x, state.y - 1)
        elif action == 2 and state.x < self.width and not self.is_wall(state.x + 1, state.y):
            next_state = GridWorldState(state.x + 1, state.y)
        elif action == 3 and state.x > 1 and not self.is_wall(state.x - 1, state.y):
            next_state = GridWorldState(state.x - 1, state.y)
        else:
            next_state = GridWorldState(state.x, state.y)

        if self.is_goal_state_single(next_state):
            is_terminal = True
        else:
            is_terminal = False

        return next_state, is_terminal

    def is_wall(self, x, y):
        '''
        Args:
            x (int)
            y (int)

        Returns:
            (bool): True iff (x,y) is a wall location.
        '''

        return (x, y) in self.walls

    def action_conv(self, str_a):
        if str_a == 'up':
            return 0
        elif str_a == 'down':
            return 1
        elif str_a == 'right':
            return 2
        elif str_a == 'left':
            return 3
        else:
            assert(False)

    def __str__(self):
        return self.name + "_h-" + str(self.height) + "_w-" + str(self.width)

    def __repr__(self):
        return self.__str__()

    def get_goal_locs(self):
        return self.goal_locs

    def get_lava_locs(self):
        return self.lava_locs
    
    def get_agent_num(self):
        return self.agent_num

    def visualize_policy(self, policy, agent_id):
        from simple_rl.utils import mdp_visualizer as mdpv
        from simple_rl.tasks.grid_world.grid_visualizer import _draw_state

        action_char_dict = {
            "up": u"\u2191",
            "down": u"\u2193",
            "left": u"\u2190",
            "right": u"\u2192"
        }

        mdpv.visualize_policy(self, policy, _draw_state, action_char_dict, agent_id)
        input("Press anything to quit")

    def visualize_value(self, agent_id):
        from simple_rl.utils import mdp_visualizer as mdpv
        from simple_rl.tasks.grid_world.grid_visualizer import _draw_state
        mdpv.visualize_value(self, _draw_state, agent_id)
        input("Press anything to quit")

    def visualize(self, cur_states, screen=None, filename=""):
        from simple_rl.utils import mdp_visualizer as mdpv
        from simple_rl.tasks.grid_world.grid_visualizer import _draw_state
        screen = mdpv.visualize(self, _draw_state, cur_states=cur_states, screen=screen, filename=filename)

        return screen

    def visualize_option(self, option_list, intra_policy_list, file_name):
        # option_list = [(init_agent_0, term_agent_0), (init_agent_1, term_agent_1), ......], len == agent_num
        # intra_policy_list = [intra_policy_agent_0, intra_policy_agent_1, ......]
        from simple_rl.utils import mdp_visualizer as mdpv
        from simple_rl.tasks.grid_world.grid_visualizer import _draw_option

        action_char_dict = {
            "up": u"\u2191",
            "down": u"\u2193",
            "left": u"\u2190",
            "right": u"\u2192"
        }

        mdpv.visualize_option(self, _draw_option, option_list, intra_policy_list, action_char_dict, file_name=file_name)
        # input("Press anything to quit")


def make_grid_world_from_file(file_name, randomize=False, num_goals=1, name=None, goal_num=None, slip_prob=0.0): #c
    '''
    Args:
        file_name (str)
        randomize (bool): If true, chooses a random agent location and goal location.
        num_goals (int)
        name (str)
    Returns:
        (GridWorldMDP)
    Summary:
        Builds a GridWorldMDP from a file:
            'w' --> wall
            'a' --> agent
            'g' --> goal
            '-' --> empty
    '''
    if name is None:
        name = (file_name.split("/")[-1]).split(".")[0]
        print("The name is ", name)

    wall_file = open(os.path.join(os.getcwd(), file_name))
    print("The path of the grid_world file: ", os.path.join(os.getcwd(), file_name))
    wall_lines = wall_file.readlines()

    # Get walls, agent, goal loc.
    num_rows = len(wall_lines)
    num_cols = len(wall_lines[0].strip())
    empty_cells = []
    agent_locs = []
    walls = []
    goal_locs = []
    lava_locs = []

    for i, line in enumerate(wall_lines):
        line = line.strip()
        for j, ch in enumerate(line):
            if ch == "w":
                walls.append((j + 1, num_rows - i))
            elif ch == "g":
                goal_locs.append((j + 1, num_rows - i))
            elif ch == "l":
                lava_locs.append((j + 1, num_rows - i))
            elif ch == "a":
                agent_locs.append((j + 1, num_rows - i))
            elif ch == "-":
                empty_cells.append((j + 1, num_rows - i))

    if goal_num is not None:
        goal_locs = [goal_locs[goal_num % len(goal_locs)]] # No. of goal

    if randomize:
        agent_num = len(agent_locs)
        agent_locs = []
        for _ in range(agent_num):
            agent_x, agent_y = random.choice(empty_cells)
            agent_locs.append((agent_x, agent_y))

        if len(goal_locs) == 0:
            # Sample @num_goals random goal locations.
            goal_locs = random.sample(empty_cells, num_goals)
        else:
            goal_locs = random.sample(goal_locs, num_goals)

    if len(goal_locs) == 0:
        goal_locs = [(num_cols, num_rows)]

    print("Goal list: ", goal_locs)
    print("Agent list: ", agent_locs)

    return GridWorldMDP(width=num_cols, height=num_rows, init_locs=agent_locs, goal_locs=goal_locs, lava_locs=lava_locs, walls=walls, name=name, slip_prob=slip_prob)


if __name__ == '__main__':
    grid_world = make_grid_world_from_file(file_name='txt_grids/tworoom.txt')
    print(grid_world.cur_states)
    print(grid_world.get_init_states())
    print(grid_world.get_curr_states())
    # cur_states = [GridWorldState(x=11,y=11), GridWorldState(x=10,y=7)]
    # print(grid_world.is_goal_state(cur_states))


