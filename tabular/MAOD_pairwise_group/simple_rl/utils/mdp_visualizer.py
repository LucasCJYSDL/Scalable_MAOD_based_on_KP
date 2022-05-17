# Python imports.
from __future__ import print_function
import sys
import time
try:
    import pygame
    from pygame.locals import *
    pygame.init()
    title_font = pygame.font.SysFont("CMU Serif", 48)
except ImportError:
    print("Error: pygame not installed (needed for visuals).")
    exit()


def val_to_color(val, good_col=(169, 193, 249), bad_col=(249, 193, 169)):
    '''
    Args:
        val (float)
        good_col (tuple)
        bad_col (tuple)

    Returns:
        (tuple)

    Summary:
        Smoothly interpolates between @good_col and @bad_col. That is,
        if @val is 1, we get good_col, if it's 0.5, we get a color
        halfway between the two, and so on.
    '''
    # Make sure val is in the appropriate range.
    val = max(min(1.0, val), -1.0)

    if val > 0:
        # Show positive as interpolated between white (0) and good_cal (1.0)
        result = tuple([255 * (1 - val) + col * val for col in good_col])
    else:
        # Show negative as interpolated between white (0) and bad_col (-1.0)
        result = tuple([255 * (1 - abs(val)) + col * abs(val) for col in bad_col])

    return result

def _draw_title_text(mdp, screen):
    '''
    Args:
        mdp (simple_rl.MDP)
        screen (pygame.Surface)

    Summary:
        Draws the name of the MDP to the top of the screen.
    '''
    scr_width, scr_height = screen.get_width(), screen.get_height()
    title_text = title_font.render(str(mdp), True, (46, 49, 49))
    screen.blit(title_text, (scr_width / 2.0 - len(str(mdp))*6, scr_width / 20.0))

def _draw_agent_text(agent, screen):
    '''
    Args:
        agent (simple_rl.Agent)
        screen (pygame.Surface)

    Summary:
        Draws the name of the agent to the bottom right of the screen.
    '''
    scr_width, scr_height = screen.get_width(), screen.get_height()
    formatted_agent_text = "agent: " + str(agent)
    agent_text_point = (3*scr_width / 4.0 - len(formatted_agent_text)*6, 18*scr_height / 20.0)
    agent_text = title_font.render(formatted_agent_text, True, (46, 49, 49))
    screen.blit(agent_text, agent_text_point)


def visualize_policy(mdp, policy, draw_state, action_char_dict, agent_id, cur_states=None, scr_width=720, scr_height=720):
    '''
    Args:
        mdp (MDP)
        policy (lambda: S --> A)
        draw_state (lambda)
        action_char_dict (dict):
            Key: action
            Val: str
        cur_state (State)

    Summary:

    '''
    screen = pygame.display.set_mode((scr_width, scr_height))

    # Setup and draw initial state.
    cur_states = mdp.get_init_states() if cur_states is None else cur_states

    _vis_init(screen)
    draw_state(screen, mdp, cur_states, agent_id=agent_id, policy=policy, action_char_dict=action_char_dict, show_value=False)

def visualize_value(mdp, draw_state, agent_id, cur_states=None, scr_width=720, scr_height=720):
    '''
    Args:
        mdp (MDP)
        draw_state (State)

    Summary:
        Draws the MDP with values labeled on states.
    '''

    screen = pygame.display.set_mode((scr_width, scr_height))

    # Setup and draw initial state.
    cur_states = mdp.get_init_states() if cur_states is None else cur_states

    _vis_init(screen)
    draw_state(screen, mdp, cur_states, show_value=True, agent_id=agent_id)

def visualize(mdp, draw_state, cur_states, screen=None, scr_width=1600, scr_height=1600, filename=""):
    '''
    Args:
        mdp (MDP)
        draw_state (lambda: State --> pygame.Rect)

    Summary:
        Creates a 2d visual of the agent's interactions with the MDP.
    '''
    if screen is None:
        screen = pygame.display.set_mode((scr_width, scr_height))

    # Setup and draw initial state.
    assert cur_states is not None

    _vis_init(screen)
    draw_state(screen, mdp, cur_states)
    pygame.image.save(screen, filename + '.png')
    # exit()
    return screen

def visualize_option(mdp, draw_option, option_list, intra_policy_list, action_char_dict, scr_width=720, scr_height=720, file_name=""):
    '''
    Args:
        mdp (MDP)
        policy (lambda: S --> A)
        draw_state (lambda)
        action_char_dict (dict):
            Key: action
            Val: str
        cur_state (State)

    Summary:

    '''
    screen = pygame.display.set_mode((scr_width, scr_height))
    _vis_init(screen)
    draw_option(screen, mdp, action_char_dict=action_char_dict, option_list=option_list, intra_policy_list=intra_policy_list)
    pygame.image.save(screen, file_name + '.png')
    # exit()
        
def _vis_init(screen):
    # Pygame setup.
    pygame.init()
    screen.fill((255, 255, 255))
    pygame.display.update()

def convert_x_y_to_grid_cell(x, y, scr_width, scr_height, mdp_width, mdp_height):
    '''
    Args:
        x (int)
        y (int)
        scr_width (int)
        scr_height (int)
        num
    '''
    width_buffer = scr_width / 10.0
    height_buffer = 30 + (scr_height / 10.0) # Add 30 for title.

    lower_left_x, lower_left_y = x - width_buffer, scr_height - y - height_buffer
    
    cell_width = (scr_width - width_buffer * 2) / mdp_width
    cell_height = (scr_height - height_buffer * 2) / mdp_height

    cell_x, cell_y = int(lower_left_x / cell_width) + 1, int(lower_left_y / cell_height) + 1

    return cell_x, cell_y
