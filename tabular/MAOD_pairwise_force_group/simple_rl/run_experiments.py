#!/usr/bin/env python
'''
Code for running experiments where RL agents interact with an MDP.

Instructions:
    (1) Create an MDP.
    (2) Create agents.
    (3) Set experiment parameters (instances, episodes, steps).
    (4) Call run_agents_on_mdp(agents, mdp) (or the lifelong/markov game equivalents).
'''

# Python imports.
from __future__ import print_function
import sys
import numpy as np
import time
from simple_rl.utils.draw_plt import plot
from tqdm import tqdm

def run_agents_on_mdp(agents_list, agent_name_list, agent_color_list, task_name, option_type, mdp, instances=5, episodes=100, steps=200, reset_at_terminal=False, visualize=False):

    # Learn.
    value_data = {}
    step_data = {}
    agent_num = mdp.get_agent_num()
    assert len(agents_list) == len(agent_name_list)
    for idx in range(len(agents_list)):
        agents = agents_list[idx]
        agent_name = agent_name_list[idx]
        print("{} is learning!".format(agents))
        temp_values = []
        temp_steps = []
        mean_mean_value = []
        mean_mean_step = []
        # For each instance.
        for instance in range(1, instances + 1):
            print("Instance " + str(instance) + " of " + str(instances) + ".")
            # sys.stdout.flush()
            value_list, step_list = run_single_agent_on_mdp(agents, mdp, episodes, steps,
                                                            reset_at_terminal=reset_at_terminal, visualize=visualize)
            print("The mean value/step of this instance is {}/{}; ".format(np.array(value_list).mean(), np.array(step_list).mean()))
            temp_values.append(value_list)
            temp_steps.append(step_list)
            # Reset the agent.
            for group_id in range(agent_num//2):
                agents[group_id].reset()
            mdp.end_of_instance()

            mean_mean_value.append(np.array(value_list).mean())
            mean_mean_step.append(np.array(step_list).mean())

        value_data[agent_name] = temp_values
        step_data[agent_name] = temp_steps
        print(np.array(mean_mean_value).mean(), " ", np.array(mean_mean_step).mean())

    plot(value_data, 'Cumulative Reward', agent_color_list, task_name, option_type)
    plot(step_data, 'Step', agent_color_list, task_name, option_type)

def run_single_agent_on_mdp(agents, mdp, episodes, steps, reset_at_terminal=False, visualize=False):

    gamma = mdp.get_gamma()
    agent_num = mdp.get_agent_num()
    value_list, step_list = [], []
    # For each episode.
    for episode in tqdm(range(1, episodes + 1)):
        # print("Episode " + str(episode) + " of " + str(episodes) + ".")
        # sys.stdout.flush()
        value = 0.0
        # Compute initial state/reward.
        mdp.reset()
        states = mdp.get_init_states()
        # print("The initial state of the agents is {}!".format(mdp.get_curr_states()))
        if visualize:
            screen = mdp.visualize(cur_states=mdp.get_curr_states())
        reward = 0.0

        for step in range(1, steps + 1):
            action_list = []
            for group_id in range(agent_num//2):
                action_list.extend(agents[group_id].act(states, reward))
            assert len(action_list) == agent_num
            # Terminal check.
            if mdp.is_goal_state(states):
                print("How could this happen?")
                break
            # Execute in MDP.
            reward, next_states, is_terminal = mdp.execute_agent_action(action_list)
            # print("After executingthe actions {}, the new state is {}!".format(action_list, next_states))
            if visualize:
                screen = mdp.visualize(cur_states=mdp.get_curr_states(), screen=screen)
                time.sleep(0.2)
            # Track value.
            value += reward * (gamma ** step)

            states = next_states

            if is_terminal:
                assert not reset_at_terminal
                break
            # Update pointer.
            # states = next_states # original!!!

        # A final update.
        for group_id in range(agent_num // 2):
            agents[group_id].act(states, reward, is_final=True) # the high level policy must be updated, no matter whether the last option terminates in the middle or not
            # Reset the MDP, tell the agent the episode is over.
            agents[group_id].end_of_episode()

        value, step = test_single_agent_on_mdp(gamma, agents, mdp, steps, visualize)
        value_list.append(value)
        step_list.append(step)

    return value_list, step_list


def test_single_agent_on_mdp(gamma, agents, mdp, steps, visualize):
    agent_num = mdp.get_agent_num()
    value = 0.0
    # Compute initial state/reward.
    mdp.reset()
    states = mdp.get_init_states()
    # print("The initial state of the agents is {}!".format(mdp.get_curr_states()))
    if visualize:
        screen = mdp.visualize(cur_states=mdp.get_curr_states())
    reward = 0.0

    for step in range(1, steps + 1):

        action_list = []
        for group_id in range(agent_num // 2):
            action_list.extend(agents[group_id].act(states, reward, learning=False))
        assert len(action_list) == agent_num
        # Terminal check.
        if mdp.is_goal_state(states):
            print("How could this happen?")
            break
        # Execute in MDP.
        reward, next_states, is_terminal = mdp.execute_agent_action(action_list)
        # print("After executingthe actions {}, the new state is {}!".format(action_list, next_states))
        if visualize:
            screen = mdp.visualize(cur_states=mdp.get_curr_states(), screen=screen)
            time.sleep(0.2)
        # Track value.
        value += reward * (gamma ** step)

        states = next_states

        if is_terminal:
            break
        # Update pointer.
        # states = next_states # original!!!

    # A final update.
    # agents.act(states, reward, is_final=True)  # the high level policy must be updated, no matter whether the last option terminates in the middle or not
    # Reset the MDP, tell the agent the episode is over.
    for group_id in range(agent_num//2):
        agents[group_id].end_of_episode()

    return value, step

