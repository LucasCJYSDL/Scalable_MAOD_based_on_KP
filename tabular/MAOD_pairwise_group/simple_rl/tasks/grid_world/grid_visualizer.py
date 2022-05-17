# Python imports.
from __future__ import print_function
from collections import defaultdict
try:
    import pygame
except ImportError:
    print("Warning: pygame not installed (needed for visuals).")

# Other imports.
from simple_rl.planning import ValueIteration
from simple_rl.utils import mdp_visualizer as mdpv
import numpy as np


def _draw_state(screen,
                grid_mdp,
                states,
                policy=None,
                action_char_dict={},
                show_value=False,
                agent_id=None):
    '''
    Returns:
        (pygame.Shape)
    '''

    agent_locs = []
    for i in range(len(states)):
        agent_loc = (states[i].x, states[i].y)
        agent_locs.append(agent_loc)

    # Make value dict.
    val_text_dict = defaultdict(lambda : defaultdict(float))
    if show_value:
        # if agent is not None:
        #     # Use agent value estimates.
        #     for s in agent.q_func.keys():
        #         val_text_dict[s.x][s.y] = agent.get_value(s)
        # else:
            # Use Value Iteration to compute value.
        assert agent_id is not None
        agent_locs = [(states[agent_id].x, states[agent_id].y)]

        vi = ValueIteration(grid_mdp, agent_id=agent_id)
        vi.run_vi()
        for s in vi.get_states():
            val_text_dict[s.x][s.y] = vi.get_value(s)

        vi.print_value_func()

    # Make policy dict.
    policy_dict = defaultdict(lambda : defaultdict(str))
    if policy:
        assert agent_id is not None
        agent_locs = [(states[agent_id].x, states[agent_id].y)]

        vi = ValueIteration(grid_mdp, agent_id) # ??
        reachable_states = vi.get_states()
        for s in reachable_states:
            policy_dict[s.x][s.y] = policy(s)

    # Prep some dimensions to make drawing easier.
    scr_width, scr_height = screen.get_width(), screen.get_height()
    width_buffer = scr_width / 10.0 # buffer: surrounding size
    height_buffer = 30 + (scr_height / 10.0) # Add 30 for title.
    cell_width = (scr_width - width_buffer * 2) / grid_mdp.width
    cell_height = (scr_height - height_buffer * 2) / grid_mdp.height
    goal_1_locs = grid_mdp.get_goal_1_locs()
    goal_2_locs = grid_mdp.get_goal_2_locs()
    goal_3_locs = grid_mdp.get_goal_3_locs()
    lava_locs = grid_mdp.get_lava_locs()

    font_size = int(min(cell_width, cell_height) / 4.0)
    reg_font = pygame.font.SysFont("CMU Serif", font_size)
    cc_font = pygame.font.SysFont("Courier", font_size*2 + 2)


    # new_agent_shapes = []
    # For each row:
    for i in range(grid_mdp.width):
        # For each column:
        for j in range(grid_mdp.height):

            top_left_point = width_buffer + cell_width*i, height_buffer + cell_height*j
            r = pygame.draw.rect(screen, (46, 49, 49), top_left_point + (cell_width, cell_height), 3)

            if show_value and not grid_mdp.is_wall(i+1, grid_mdp.height - j):
                # Draw the value.
                val = (val_text_dict[i+1][grid_mdp.height - j] - 0.9) * 10
                # val = val_text_dict[i + 1][grid_mdp.height - j]
                color = mdpv.val_to_color(val)
                pygame.draw.rect(screen, color, top_left_point + (cell_width, cell_height), 0)

            if grid_mdp.is_wall(i+1, grid_mdp.height - j):
                # Draw the walls.
                top_left_point = width_buffer + cell_width*i + 5, height_buffer + cell_height*j + 5
                r = pygame.draw.rect(screen, (94, 99, 99), top_left_point + (cell_width-10, cell_height-10), 0)

            if (i + 1, grid_mdp.height - j) in goal_1_locs:
                # Draw goal.
                # TODO: Better visualization?
                circle_center = int(top_left_point[0] + cell_width / 2.0), int(top_left_point[1] + cell_height / 2.0)
                circle_color = (154, 195, 157)
                pygame.draw.circle(screen, circle_color, circle_center, int(min(cell_width, cell_height) / 2.3))
                # pass

            if (i + 1, grid_mdp.height - j) in goal_2_locs:
                # Draw goal.
                # TODO: Better visualization?
                circle_center = int(top_left_point[0] + cell_width / 2.0), int(top_left_point[1] + cell_height / 2.0)
                circle_color = (195, 154, 157)
                pygame.draw.circle(screen, circle_color, circle_center, int(min(cell_width, cell_height) / 2.3))
                # pass

            if (i + 1, grid_mdp.height - j) in goal_3_locs:
                # Draw goal.
                # TODO: Better visualization?
                circle_center = int(top_left_point[0] + cell_width / 2.0), int(top_left_point[1] + cell_height / 2.0)
                circle_color = (154, 157, 195)
                pygame.draw.circle(screen, circle_color, circle_center, int(min(cell_width, cell_height) / 2.3))

            if (i+1,grid_mdp.height - j) in lava_locs:
                # Draw goal.
                circle_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)
                circler_color = (224, 145, 157)
                pygame.draw.circle(screen, circler_color, circle_center, int(min(cell_width, cell_height) / 4.0))

            # Current state.
            if (i+1,grid_mdp.height - j) in agent_locs:
                group_id = (agent_locs.index((i+1,grid_mdp.height - j)))//2
                if group_id == 0:
                    tri_color = (154, 195, 157)
                elif group_id == 1:
                    tri_color = (195, 154, 157)
                else:
                    tri_color = (154, 157, 195)
                tri_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)
                _draw_agent(tri_center, screen, base_size=min(cell_width, cell_height)/2.3 - 3, tri_color=tri_color)
                # new_agent_shapes.append(agent_shape)

            if policy and not grid_mdp.is_wall(i+1, grid_mdp.height - j): # the axies of the screen is different from the matrix's
                a = policy_dict[i+1][grid_mdp.height - j]
                if a not in action_char_dict:
                    text_a = a
                else:
                    text_a = action_char_dict[a]
                text_center_point = int(top_left_point[0] + cell_width/2.0 - 10), int(top_left_point[1] + cell_height/3.0)
                text_rendered_a = cc_font.render(text_a, True, (46, 49, 49))
                screen.blit(text_rendered_a, text_center_point)

    pygame.display.flip()


def _draw_agent(center_point, screen, base_size=20, tri_color=None):
    '''
    Args:
        center_point (tuple): (x,y)
        screen (pygame.Surface)

    Returns:
        (pygame.rect)
    '''
    tri_bot_left = center_point[0] - base_size, center_point[1] + base_size
    tri_bot_right = center_point[0] + base_size, center_point[1] + base_size
    tri_top = center_point[0], center_point[1] - base_size
    tri = [tri_bot_left, tri_top, tri_bot_right]
    if tri_color is None:
        tri_color = (98, 140, 190)
    return pygame.draw.polygon(screen, tri_color, tri)


def _draw_option(screen,
                grid_mdp,
                action_char_dict={},
                option_list = [],
                intra_policy_list = []):
    # option_list = [(init_agent_0, term_agent_0), (init_agent_1, term_agent_1), ......], len == agent_num
    # intra_policy_list = [intra_policy_agent_0, intra_policy_agent_1, ......]
    '''
    Returns:
        (pygame.Shape)
    '''

    # Prep some dimensions to make drawing easier.
    scr_width, scr_height = screen.get_width(), screen.get_height()
    width_buffer = scr_width / 10.0 # buffer: surrounding size
    height_buffer = 30 + (scr_height / 10.0) # Add 30 for title.
    cell_width = (scr_width - width_buffer * 2) / grid_mdp.width
    cell_height = (scr_height - height_buffer * 2) / grid_mdp.height
    goal_1_locs = grid_mdp.get_goal_1_locs()
    goal_2_locs = grid_mdp.get_goal_2_locs()
    goal_3_locs = grid_mdp.get_goal_3_locs()
    lava_locs = grid_mdp.get_lava_locs()

    font_size = int(min(cell_width, cell_height) / 4.0)
    reg_font = pygame.font.SysFont("CMU Serif", font_size)
    cc_font = pygame.font.SysFont("Courier", font_size*2 + 2)

    # For each row:
    for i in range(grid_mdp.width):
        # For each column:
        for j in range(grid_mdp.height):

            top_left_point = width_buffer + cell_width*i, height_buffer + cell_height*j
            r = pygame.draw.rect(screen, (46, 49, 49), top_left_point + (cell_width, cell_height), 3)

            if grid_mdp.is_wall(i+1, grid_mdp.height - j):
                # Draw the walls.
                top_left_point = width_buffer + cell_width*i + 5, height_buffer + cell_height*j + 5
                r = pygame.draw.rect(screen, (94, 99, 99), top_left_point + (cell_width-10, cell_height-10), 0)

            if (i+1,grid_mdp.height - j) in goal_1_locs:
                # Draw goal.
                # TODO: Better visualization?
                circle_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)
                circle_color = (154, 195, 157)
                pygame.draw.circle(screen, circle_color, circle_center, int(min(cell_width, cell_height) / 3.0))
                # pass

            if (i+1,grid_mdp.height - j) in goal_2_locs:
                # Draw goal.
                # TODO: Better visualization?
                circle_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)
                circle_color = (195, 154, 157)
                pygame.draw.circle(screen, circle_color, circle_center, int(min(cell_width, cell_height) / 3.0))
                # pass

            if (i+1,grid_mdp.height - j) in goal_3_locs:
                # Draw goal.
                # TODO: Better visualization?
                circle_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)
                circle_color = (154, 157, 195)
                pygame.draw.circle(screen, circle_color, circle_center, int(min(cell_width, cell_height) / 3.0))
                # pass

            if (i+1,grid_mdp.height - j) in lava_locs:
                # Draw goal.
                circle_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)
                circler_color = (224, 145, 157)
                pygame.draw.circle(screen, circler_color, circle_center, int(min(cell_width, cell_height) / 4.0))

    agent_num = len(option_list)
    assert len(intra_policy_list) == agent_num
    for agent_id in range(agent_num):
        temp_option = option_list[agent_id]
        temp_policy = intra_policy_list[agent_id]
        temp_color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        # draw agent_loc
        for i in range(2):
            agent_loc = (temp_option[i].x, temp_option[i].y)
            top_left_point = (width_buffer + cell_width * (agent_loc[0] - 1), height_buffer + cell_height * (grid_mdp.height - agent_loc[1]))
            tri_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)
            _draw_agent(tri_center, screen, base_size=min(cell_width, cell_height) / 2.5 - 8, tri_color=temp_color)
        # draw intra-option policy
        # init_loc = (temp_option[0].x, temp_option[0].y)
        # term_loc = (temp_option[1].x, temp_option[1].y)
        temp_state = temp_option[0]
        # print("1: ", temp_option[1])
        while not (temp_state == temp_option[1]):
            # print("2: ", temp_state)
            action = temp_policy(temp_state)
            text_a = action_char_dict[action]
            temp_loc = (temp_state.x, temp_state.y)
            top_left_point = (width_buffer + cell_width * (temp_loc[0] - 1), height_buffer + cell_height * (grid_mdp.height - temp_loc[1]))
            text_center_point = int(top_left_point[0] + cell_width / 2.0 - 10), int(top_left_point[1] + cell_height / 3.0)
            text_rendered_a = cc_font.render(text_a, True, temp_color)
            screen.blit(text_rendered_a, text_center_point)
            temp_state, _ = grid_mdp._transition_func_single(temp_state, action, agent_id, allow_further=True)

    pygame.display.flip()