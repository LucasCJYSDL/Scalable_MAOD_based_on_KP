import os
from copy import copy
# Other imports.
from simple_rl.planning.ValueIterationClass import ValueIteration
from simple_rl.abstraction.action_abs.InListPredicateClass import InListPredicate
from simple_rl.abstraction.action_abs.OptionClass import Option
from simple_rl.abstraction.action_abs.PolicyFromDictClass import PolicyFromDict
from simple_rl.abstraction.action_abs.IntrinsitcMDP import IntrinsicMDP
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState

def make_point_options(mdp, option_list, intToS_list, policy, name, visualize_option=False):
    assert policy == 'vi'
    if 'single' in name:
        # option_list: [[(init_0_agent_0, term_0_agent_0), ((init_1_agent_0, term_1_agent_0))......], \
        #               [(init_0_agent_1, term_0_agent_1), ((init_1_agent_1, term_1_agent_1))......]]
        agent_num = len(option_list)
        option_class_list = [[] for _ in range(agent_num)]
        for agent_id in range(agent_num):
            pairs = option_list[agent_id]
            option_visual_list = []
            policy_visual_list = []

            for pair in pairs:

                init = intToS_list[agent_id][pair[0]] # GridWorldState
                term = intToS_list[agent_id][pair[1]]
                mdp_ = IntrinsicMDP(intrinsic_term_list=[term], mdp=mdp)
                o = Option(init_predicate=InListPredicate(ls=[init]),
                           term_predicate=InListPredicate(ls=[term]),
                           policy=_make_mini_mdp_option_policy(mdp_, agent_id),
                           term_prob=0.0)
                option_class_list[agent_id].append(o)

                # not included in the Jinnai's code
                mdp_ = IntrinsicMDP(intrinsic_term_list=[init], mdp=mdp)
                o_reverse = Option(init_predicate=InListPredicate(ls=[term]),
                                   term_predicate=InListPredicate(ls=[init]),
                                   policy=_make_mini_mdp_option_policy(mdp_, agent_id),
                                   term_prob=0.0)
                option_class_list[agent_id].append(o_reverse)

                option_visual_list.append((init, term))
                policy_visual_list.append(o.policy)

            if visualize_option:
                if not os.path.exists('../visualization'):
                    os.makedirs('../visualization')
                mdp.visualize_option(option_visual_list, policy_visual_list, file_name="../visualization/single_option_agent_{}".format(agent_id))

    else:
        assert 'multiple' in name
        # multi_option_list: [ [[(agent_0_init, agent_1_init), (agent_0_term, agent_1_term)], ......],
        #                      [[(agent_2_init, agent_3_init), (agent_2_term, agent_3_term)], ......] ]

        group_num = len(option_list)
        option_class_list = [[] for _ in range(group_num)]

        for group_id in range(group_num):
            option_num = len(option_list[group_id])
            assert option_num > 0
            agent_num = len(option_list[group_id][0][0])
            assert agent_num == 2

            for i in range(option_num):
                option_i = option_list[group_id][i]
                init_funcs = []
                term_funcs = []
                intra_policy_list = []
                intra_policy_rev_list = []
                option_visual_list = []
                policy_visual_list = []
                for agent_id in range(agent_num):
                    intToS_idx = agent_num * group_id + agent_id
                    init = intToS_list[intToS_idx][option_i[0][agent_id]]
                    term = intToS_list[intToS_idx][option_i[1][agent_id]]
                    assert isinstance(init, GridWorldState) and isinstance(term, GridWorldState)
                    init_funcs.append(InListPredicate(ls=[init]))
                    term_funcs.append(InListPredicate(ls=[term]))

                    mdp_ = IntrinsicMDP(intrinsic_term_list=[term], mdp=mdp)
                    intra_policy = _make_mini_mdp_option_policy(mdp_, agent_id=intToS_idx)
                    intra_policy_list.append(intra_policy)

                    mdp_rev = IntrinsicMDP(intrinsic_term_list=[init], mdp=mdp)
                    intra_policy_rev = _make_mini_mdp_option_policy(mini_mdp=mdp_rev, agent_id=intToS_idx)
                    intra_policy_rev_list.append(intra_policy_rev)

                    option_visual_list.append((init, term))
                    policy_visual_list.append(intra_policy)

                if visualize_option:
                    if not os.path.exists('../visualization'):
                        os.makedirs('../visualization')
                    mdp.visualize_option(option_visual_list, policy_visual_list, file_name="../visualization/multiple_option_{}_group_{}".format(i, group_id))

                o = Option(init_predicate=init_funcs,
                           term_predicate=term_funcs,
                           policy=intra_policy_list,
                           term_prob=0.0, is_single=False, group_id=group_id)
                option_class_list[group_id].append(o)

                o_reverse = Option(init_predicate=term_funcs,
                                   term_predicate=init_funcs,
                                   policy=intra_policy_rev_list,
                                   term_prob=0.0, is_single=False, group_id=group_id)
                option_class_list[group_id].append(o_reverse)

    return option_class_list


def make_subgoal_options(mdp, option_list, partition_list, intToS_list, policy, name):
    assert policy == 'vi'
    if 'single' in name:
        # option_list: [[(init_0_agent_0, term_0_agent_0), ((init_1_agent_0, term_1_agent_0))......], \
        #               [(init_0_agent_1, term_0_agent_1), ((init_1_agent_1, term_1_agent_1))......]
        #               [(init_0_agent_2, term_0_agent_2), ((init_1_agent_2, term_1_agent_2))......], \
        #               [(init_0_agent_3, term_0_agent_3), ((init_1_agent_3, term_1_agent_3))......]]
        assert len(option_list) == len(partition_list)
        agent_num = len(option_list)
        option_class_list = [[] for _ in range(agent_num)]
        for agent_id in range(agent_num):
            known_region = list(intToS_list[agent_id].values())
            pairs = option_list[agent_id]
            partitions = partition_list[agent_id]
            assert len(pairs) == len(partitions)
            for idx in range(len(pairs)):
                pair = pairs[idx]
                partition = partitions[idx]
            # for pair in pairs:
                for i in range(2): # init: max_point and term: min_point
                    term = copy(known_region)
                    print("The size before remove: ", len(term))
                    assert intToS_list[agent_id][pair[i]] in term
                    print(intToS_list[agent_id][pair[i]])
                    term.remove(intToS_list[agent_id][pair[i]])
                    print("The size after remove: ", len(term))
                    mdp_ = IntrinsicMDP(intrinsic_term_list=[intToS_list[agent_id][pair[i]]], mdp=mdp) # a little waste of time

                    init = []
                    for state_idx in partition[i]:
                        init.append(intToS_list[agent_id][state_idx])

                    o = Option(init_predicate=InListPredicate(ls=init),
                               term_predicate=InListPredicate(ls=term, true_if_in=False),
                               policy=_make_mini_mdp_option_policy(mdp_, agent_id),
                               term_prob=0.0)
                    option_class_list[agent_id].append(o)

    else:
        assert 'multiple' in name
        # multi_option_list: [ [[(agent_0_init, agent_1_init), (agent_0_term, agent_1_term)], ......],
        #                      [[(agent_2_init, agent_3_init), (agent_2_term, agent_3_term)], ......] ]
        assert len(partition_list) == len(option_list)
        group_num = len(option_list)
        option_class_list = [[] for _ in range(group_num)]

        for group_id in range(group_num):
            assert len(partition_list[group_id]) == len(option_list[group_id])
            option_num = len(option_list[group_id]) # may be different for each group
            assert option_num > 0
            agent_num = len(option_list[group_id][0][0])
            assert agent_num == 2
            for i in range(option_num):
                option_i = option_list[group_id][i]
                partition_i = partition_list[group_id][i]
                # available_sign = []
                term_funcs = []
                term_rev_funcs = []
                intra_policy_list = []
                intra_policy_rev_list = []
                for agent_id in range(agent_num):
                    intToS_idx = agent_num * group_id + agent_id
                    known_region = list(intToS_list[intToS_idx].values())
                    print("The size of the known region is {}!".format(len(known_region)))
                    # init_funcs.append(InListPredicate(ls=known_region))

                    term = copy(known_region)
                    assert intToS_list[intToS_idx][option_i[1][agent_id]] in known_region
                    term.remove(intToS_list[intToS_idx][option_i[1][agent_id]]) # check
                    term_funcs.append(InListPredicate(ls=term, true_if_in=False))

                    mdp_ = IntrinsicMDP(intrinsic_term_list=[intToS_list[intToS_idx][option_i[1][agent_id]]], mdp=mdp)
                    policy = _make_mini_mdp_option_policy(mdp_, agent_id=intToS_idx)
                    intra_policy_list.append(policy)

                    term_rev = copy(known_region)
                    assert intToS_list[intToS_idx][option_i[0][agent_id]] in known_region
                    term_rev.remove(intToS_list[intToS_idx][option_i[0][agent_id]])  # check
                    term_rev_funcs.append(InListPredicate(ls=term_rev, true_if_in=False))

                    mdp_rev = IntrinsicMDP(intrinsic_term_list=[intToS_list[intToS_idx][option_i[0][agent_id]]], mdp=mdp)
                    policy_rev = _make_mini_mdp_option_policy(mdp_rev, agent_id=intToS_idx)
                    intra_policy_rev_list.append(policy_rev)

                init_dict = {}
                init_rev_dict = {}
                known_region_0 = list(intToS_list[agent_num * group_id].values())
                known_region_1 = list(intToS_list[agent_num * group_id + 1].values())
                for s_0 in known_region_0:
                    for s_1 in known_region_1:
                        init_dict[(s_0, s_1)] = False
                        init_rev_dict[(s_0, s_1)] = False

                print("1: ", len(init_dict))
                print("2: ", len(init_rev_dict))

                init_partition_i = partition_i[1]
                for state_idx in init_partition_i:
                    temp_list = []
                    for agent_idx in range(agent_num):
                        intToS_idx = agent_num * group_id + agent_idx
                        temp_list.append(intToS_list[intToS_idx][state_idx[agent_idx]])
                    init_dict[tuple(temp_list)] = True

                o = Option(init_predicate=init_dict,
                           term_predicate=term_funcs,
                           policy=intra_policy_list,
                           term_prob=0.0, is_single=False, group_id=group_id)
                option_class_list[group_id].append(o)

                init_rev_partition_i = partition_i[0]
                for state_idx in init_rev_partition_i:
                    temp_list = []
                    for agent_idx in range(agent_num):
                        intToS_idx = agent_num * group_id + agent_idx
                        temp_list.append(intToS_list[intToS_idx][state_idx[agent_idx]])
                    init_rev_dict[tuple(temp_list)] = True

                o_rev = Option(init_predicate=init_rev_dict,
                               term_predicate=term_rev_funcs,
                               policy=intra_policy_rev_list,
                               term_prob=0.0, is_single=False, group_id=group_id)
                option_class_list[group_id].append(o_rev)

    return option_class_list



def _make_mini_mdp_option_policy(mini_mdp, agent_id):
    '''
    Args:
        mini_mdp (MDP)
    Returns:
        Policy
    '''
    # Solve the MDP defined by the terminal abstract state.
    mini_mdp_vi = ValueIteration(mini_mdp, agent_id=agent_id) # the vi results induced by different agents are the same
    mini_mdp_vi.run_vi()

    o_policy_dict = make_dict_from_lambda(mini_mdp_vi.policy, mini_mdp_vi.get_states())
    o_policy = PolicyFromDict(o_policy_dict)

    # print('type(policy)=', type(o_policy.get_action))

    return o_policy.get_action

def make_dict_from_lambda(policy_func, state_list):
    policy_dict = {}
    for s in state_list:
        policy_dict[s] = policy_func(s)

    return policy_dict
