#!/usr/bin/env python
# Python imports.
import os
from datetime import datetime
import argparse
# simple_rl
from simple_rl.tasks.grid_world.GridWorldMDPClass import make_grid_world_from_file
from simple_rl.agents import CentQLearningAgent, MultiCentQLearningAgent
from simple_rl.abstraction import AbstractionWrapper, aa_helpers, ActionAbstraction
from simple_rl.run_experiments import run_agents_on_mdp
# options
from options.option_generation.util import GetAdjacencyMatrix, GetIncidenceMatrix
from options.option_generation.multiple_options import multi_options
from options.option_generation.single_options import single_options


def GetOption(mdp, k=1, num_limit=2, matrix_list=None, intToS=None, id_list=None, method='multiple', normalized=True, with_degree=True, use_median=False):
    assert matrix_list is not None
    A_list = matrix_list

    if method == 'multiple':
        adj_list, option_list, option_num, partition_list = multi_options(mdp, A_list, intToS, id_list, k, num_limit, normalized=normalized,
                                                                          with_degree=with_degree, use_median=use_median)
        # option_list: [[(agent_0_init, agent_1_init), (agent_0_term, agent_1_term)], ......]
    else:
        assert method == 'single'
        adj_list, option_list, option_num, partition_list = single_options(A_list, k, with_degree, use_median)
        # option_list: [[(init_0_agent_0, term_0_agent_0), ((init_1_agent_0, term_1_agent_0))......], \
        #               [(init_0_agent_1, term_0_agent_1), ((init_1_agent_1, term_1_agent_1))......]
        #               [(init_0_agent_2, term_0_agent_2), ((init_1_agent_2, term_1_agent_2))......], \
        #               [(init_0_agent_3, term_0_agent_3), ((init_1_agent_3, term_1_agent_3))......],.....]
    print("Total number of options: ", option_num)
    print("The option list: ", option_list)
    return adj_list, option_list, partition_list


def build_option_agent(mdp, option_list, partition_list, intToS_list, agent_num, agent_class, option_type, policy='vi', name='-abstr'):
    if option_type == 'subgoal':
        goal_based_options = aa_helpers.make_subgoal_options(mdp, option_list, partition_list, intToS_list, policy=policy, name=name)  # option (low-level policy)
    else:
        goal_based_options = aa_helpers.make_point_options(mdp, option_list, intToS_list, policy=policy, name=name)
    # single_options: [[option_0_agent_0, option_1_agent_0,......], [option_0_agent_1, option_1_agent_1,......], ......]
    # multiple_options: [[multi_option_0, multi_option_1,......]_group_0, [multi_option_0, multi_option_1,......]_group_1, ......]
    # print("The length of the goal list is {}!".format(len(goal_based_options)))

    option_agent_group = []
    for group_id in range(agent_num//2):
        goal_based_aa = ActionAbstraction(prim_actions=mdp.get_actions(), options=goal_based_options, use_prims=True,
                                          name=name, gamma=mdp.get_gamma(), agent_num=2, group_id=group_id) # (high-level policy)
        option_agent = AbstractionWrapper(SubAgentClass=agent_class, agent_params={"avai_action_list": None, "agent_num": 2, 'group_id': group_id},
                                          action_abstr=goal_based_aa, name_ext=name)
        option_agent_group.append(option_agent)

    return option_agent_group


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
    
    n_options = args.noptions
    option_type = args.optiontype
    element_limit = args.element_limit
    normalized = args.normalized
    with_degree = args.with_degree
    use_median = args.use_median

    agent_class = CentQLearningAgent
    multi_agent_class = MultiCentQLearningAgent

    agent_num = mdp.get_agent_num()
    assert agent_num % 2 == 0
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
        print("The state space size of Agent {} is {}!".format(agent_id, len(list(intToS.values()))))

    multi_matrix_list = []
    multi_option_list = []
    multi_partition_list = []
    for idx in range(agent_num // 2):
        pair_multi_matrix, pair_multi_option, pair_multi_partition = GetOption(mdp, k=n_options, intToS=intToS_list[(idx * 2):(idx * 2 + 2)], id_list=[(idx * 2), (idx * 2)+1], num_limit=element_limit,
                                                         matrix_list=origMatrix_list[(idx * 2):(idx * 2 + 2)],
                                                         normalized=normalized, method='multiple', with_degree=with_degree, use_median=use_median)
        multi_matrix_list.append(pair_multi_matrix)
        multi_option_list.append(pair_multi_option)
        multi_partition_list.append(pair_multi_partition)
    # multi_option_list: [ [[(agent_0_init, agent_1_init), (agent_0_term, agent_1_term)], ......],
    #                      [[(agent_2_init, agent_3_init), (agent_2_term, agent_3_term)], ......], ...... ]
    multi_option_num = len(multi_option_list[0])

    single_matrix_list, single_option_list, single_partition_list = GetOption(mdp, k=int(multi_option_num**0.5), matrix_list=origMatrix_list, method='single', with_degree=with_degree, use_median=use_median)

   # #################################
   #  # Subgoal options
   #  #################################
    multi_option_agents = build_option_agent(mdp, multi_option_list, multi_partition_list, intToS_list, agent_num, agent_class=multi_agent_class,
                                            option_type=option_type, policy='vi', name='-multiple')
    single_option_agents = build_option_agent(mdp, single_option_list,single_partition_list, intToS_list, agent_num, agent_class=agent_class,
                                             option_type=option_type, policy='vi', name='-single')
   #
    cent_agents = []
    for group_id in range(agent_num//2):
        cent_agent = CentQLearningAgent(avai_action_list=[mdp.get_actions(), mdp.get_actions()], agent_num=2, group_id=group_id)
        cent_agents.append(cent_agent)
    # [multi_option_agents, single_option_agents, cent_agents]
    # [args.agent+'_multi_option', args.agent+'_single_option', 'CentQ']
    # ['blue', 'red', 'orange']
    run_agents_on_mdp([multi_option_agents, single_option_agents, cent_agents],
                      [args.agent+'_multi_option', args.agent+'_single_option', 'CentQ'],
                      ['blue', 'red', 'orange'], args.task, option_type + '_group',
                      mdp, instances=n_instances,
                      episodes=n_episodes, steps=n_steps, reset_at_terminal=False, visualize=visualize)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment', type=str, default='offline', help='online, offline')
    parser.add_argument('--incidence', type=bool, default=False)
    parser.add_argument('--normalized', type=bool, default=True)
    parser.add_argument('--with_degree', type=bool, default=False)
    parser.add_argument('--use_median', type=bool, default=True)
    parser.add_argument('--optiontype', type=str, default='subgoal') # 'point' or 'subgoal'
    # Parameters for the task
    parser.add_argument('--task', type=str, default='grid_room4') # grid_fourroom, grid_tworoom
    parser.add_argument('--nepisodes', type=int, default=1000)
    parser.add_argument('--nsteps', type=int, default=200)
    parser.add_argument('--ninstances', type=int, default=5)
    parser.add_argument('--visualize', type=bool, default=False)
    # Parameters for the high-level algorithm
    parser.add_argument('--agent', type=str, default='CentQ_force')

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

