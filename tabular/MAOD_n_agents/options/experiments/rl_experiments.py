#!/usr/bin/env python
# Python imports.
import os
from datetime import datetime
import argparse
# simple_rl
from simple_rl.tasks.grid_world.GridWorldMDPClass import make_grid_world_from_file
from simple_rl.agents import QLearningAgent, RandomAgent, DistQLearningAgent
from simple_rl.abstraction import AbstractionWrapper, aa_helpers, ActionAbstraction
from simple_rl.run_experiments import run_agents_on_mdp
# options
from options.option_generation.util import GetAdjacencyMatrix, GetIncidenceMatrix
from options.option_generation.multiple_options import multi_options
from options.option_generation.single_options import single_options


def GetOption(mdp, k=1, num_limit=2, matrix_list=None, intToS=None, method='multiple', normalized=True, with_degree=True, use_median=False):
    assert matrix_list is not None
    A_list = matrix_list

    if method == 'multiple':
        adj_list, option_list, option_num, partition_list = multi_options(mdp, A_list, intToS, k, num_limit, normalized=normalized, with_degree=with_degree, use_median=use_median)
        # option_list: [[(agent_0_init, agent_1_init), (agent_0_term, agent_1_term)], ......]
        # partition_list = [[[(agent_0_min_0, agent_1_min_0), (agent_0_min_1, agent_1_min_1), ...],
        #                    [(agent_0_max_0, agent_1_max_0), (agent_0_max_1, agent_1_max_1), ...]], ......]
    else:
        assert method == 'single'
        adj_list, option_list, option_num, partition_list = single_options(A_list, k, with_degree, use_median)
        # option_list: [[(init_0_agent_0, term_0_agent_0), (init_1_agent_0, term_1_agent_0), ......], \
        #               [(init_0_agent_1, term_0_agent_1), (init_1_agent_1, term_1_agent_1), ......]]
        # partition_list: [[([agent_0_min_0, agent_0_min_1, ...], [agent_0_max_0, agent_0_max_1, ...]), ...],
        #                  [([agent_1_min_0, agent_1_min_1, ...], [agent_1_max_0, agent_1_max_1, ...]), ...]]
    print("Total number of options: ", option_num)
    print("The option list: ", option_list)
    return adj_list, option_list, partition_list


def build_option_agent(mdp, option_list, partition_list, intToS_list, agent_num, agent_class, option_type, policy='vi', name='-abstr', visualize_option=False):
    if option_type == 'subgoal':
        goal_based_options = aa_helpers.make_subgoal_options(mdp, option_list, partition_list, intToS_list, policy=policy, name=name)  # option (low-level policy)
    else:
        goal_based_options = aa_helpers.make_point_options(mdp, option_list, intToS_list, policy=policy, name=name, visualize_option=visualize_option)
    # single_options: [[option_0_agent_0, option_1_agent_0,......], [option_0_agent_1, option_1_agent_1,......]]
    # multiple_options: [multi_option_0, multi_option_1,......]
    # if 'single' in name:
    #     option_list_len = len(goal_based_options[0])
    # else:
    #     option_list_len = len(goal_based_options)

    print("The length of the goal list is {}!".format(len(goal_based_options)))
    option_agent_list = []
    for agent_id in range(agent_num):
        goal_based_aa = ActionAbstraction(prim_actions=mdp.get_actions(), options=goal_based_options, use_prims=True,
                                          name=name, agent_id=agent_id, gamma=mdp.get_gamma()) # (high-level policy)
        option_agent = AbstractionWrapper(SubAgentClass=agent_class, agent_params={"actions": mdp.get_actions(), "agent_id": agent_id},
                                          action_abstr=goal_based_aa, name_ext=name, agent_id=agent_id)
        option_agent_list.append(option_agent)

    return option_agent_list


def test_offline_agent(args, mdp):
    '''
    '''
    #########################
    # Parameters for the Offline option generations
    # Incidence matrix sampling
    smp_n_traj = args.nsepisodes
    smp_steps = args.nssteps
    
    # Final Evaluation step
    n_episodes = args.nepisodes
    n_steps = args.nsteps
    n_instances = args.ninstances
    visualize = args.visualize
    visualize_option = args.visualize_option
    
    n_options = args.noptions
    option_type = args.optiontype
    element_limit = args.element_limit
    normalized = args.normalized
    with_degree = args.with_degree
    use_median = args.use_median

    if args.agent == 'Q':
        agent_class = QLearningAgent
    elif args.agent == 'DistQ':
        agent_class = DistQLearningAgent
    else:
        agent_class = RandomAgent

    agent_num = mdp.get_agent_num()
    print("The agent number is {}!!!!!!".format(agent_num))

    # now = datetime.now()
    # now_ts = str(now.timestamp())
    origMatrix_list = []
    intToS_list = []
    for agent_id in range(agent_num):
        if args.incidence:
            origMatrix, intToS = GetIncidenceMatrix(mdp, agent_id=agent_id, n_traj=smp_n_traj, eps_len=smp_steps) # based on the sampled states
        else:
            origMatrix, intToS = GetAdjacencyMatrix(mdp, agent_id=agent_id) # based on all the reachable states
        origMatrix_list.append(origMatrix)
        intToS_list.append(intToS)
        # print("The state transition graph of Agent {} is:".format(agent_id))
        # for row in origMatrix:
        #     print(row)
        print("The state space of Agent {} is {}!".format(agent_id, intToS))

    multi_matrix_list, multi_option_list, multi_partition_list = GetOption(mdp, k=n_options, intToS=intToS_list ,num_limit=element_limit,
                                                         matrix_list=origMatrix_list, normalized=normalized, with_degree=with_degree, method='multiple', use_median=use_median)
    multi_option_num = len(multi_option_list)
    single_matrix_list, single_option_list, single_partition_list = GetOption(mdp, k=int(multi_option_num ** 0.5), with_degree=with_degree,
                                                            matrix_list=origMatrix_list, method='single', use_median=use_median)

   # #################################
   #  # Subgoal options
   #  #################################
    multi_option_agents = build_option_agent(mdp, multi_option_list, multi_partition_list, intToS_list, agent_num, agent_class=agent_class,
                                             option_type=option_type, policy='vi', name='-multiple')
    single_option_agents = build_option_agent(mdp, single_option_list, single_partition_list, intToS_list, agent_num, agent_class=agent_class,
                                              option_type=option_type, policy='vi', name='-single')

   #
    ql_agents = []
    rand_agents = []
    dql_agents = []
    for agent_id in range(agent_num):
        # ql_agent = QLearningAgent(actions=mdp.get_actions(), agent_id=agent_id, default_q=1.0)
        ql_agent = QLearningAgent(actions=mdp.get_actions(), agent_id=agent_id)
        rand_agent = RandomAgent(actions=mdp.get_actions(), agent_id=agent_id)
        dql_agent = DistQLearningAgent(actions=mdp.get_actions(), agent_id=agent_id)
        ql_agents.append(ql_agent)
        rand_agents.append(rand_agent)
        dql_agents.append(dql_agent)
    # [multi_option_agents, single_option_agents, ql_agents, rand_agents, dql_agents]
    # [args.agent+'_multi_option', args.agent+'_single_option', 'Q', 'Random', 'DistQ']
    # ['blue', 'red', 'orange', 'black', 'green']
    run_agents_on_mdp([multi_option_agents, single_option_agents, dql_agents],
                      [args.agent+'_multi_option', args.agent+'_single_option', 'DistQ'],
                      ['blue', 'red', 'green'], args.task + '_' + option_type + '_' + args.agent,
                      mdp, instances=n_instances, episodes=n_episodes, steps=n_steps, reset_at_terminal=False, visualize=visualize)

    # run_agents_on_mdp([multi_option_agents, single_option_agents, ql_agents],
    #                   [args.agent + '_multiple_agent_option', args.agent + '_single_agent_option', 'Q'],
    #                   ['blue', 'red', 'orange'], args.task + '_' + option_type + '_' + args.agent,
    #                   mdp, instances=n_instances, episodes=n_episodes, steps=n_steps, reset_at_terminal=False,
    #                   visualize=visualize)

    # run_agents_on_mdp([multi_option_agents, single_option_agents, rand_agents],
    #                   [args.agent + '_multiple_agent_option', args.agent + '_single_agent_option', 'Random'],
    #                   ['blue', 'red', 'black'], args.task + '_' + option_type + '_' + args.agent,
    #                   mdp, instances=n_instances, episodes=n_episodes, steps=n_steps, reset_at_terminal=False,
    #                   visualize=visualize)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment', type=str, default='offline', help='online, offline')
    parser.add_argument('--incidence', type=bool, default=False)
    parser.add_argument('--normalized', type=bool, default=True)
    parser.add_argument('--with_degree', type=bool, default=False)
    parser.add_argument('--use_median', type=bool, default=False)
    parser.add_argument('--optiontype', type=str, default='subgoal') # 'point' or 'subgoal'
    # Parameters for the task
    parser.add_argument('--task', type=str, default='grid_room4') # grid_fourroom, grid_tworoom
    parser.add_argument('--nepisodes', type=int, default=1000)
    parser.add_argument('--nsteps', type=int, default=200)
    parser.add_argument('--ninstances', type=int, default=5)
    parser.add_argument('--visualize', type=bool, default=False)
    parser.add_argument('--visualize_option', type=bool, default=False)
    # Parameters for the high-level algorithm
    parser.add_argument('--agent', type=str, default='Q') # Q, Random, DistQ

    # Parameters for *offline* ODQN
    parser.add_argument('--nsepisodes', type=int, default=200, help='number of episodes for incidence matrix sampling')
    parser.add_argument('--nssteps', type=int, default=400, help='number of steps for incidence matrix sampling')
    parser.add_argument('--noptions', type=int, default=1)
    parser.add_argument('--element_limit', type=int, default=2)

    args = parser.parse_args()

    dom, task = args.task.split('_')
    
    if dom == 'grid':
        file_name = os.path.dirname(os.path.realpath(__file__)) + '/../tasks/' + task + '.txt'
        print("Grid World file path: ", file_name)
        mdp = make_grid_world_from_file(file_name=file_name)
    else:
        print('Unknown task name: ', task)
        assert(False)

    mdp.set_gamma(0.99)

    print("The parameters of the env: ", mdp.get_parameters())

    if args.experiment == 'offline':
        print('test_offline_agent')
        test_offline_agent(args, mdp)
    else:
        print('Unregisterd experiment:', args.experiment)
        assert(False)

