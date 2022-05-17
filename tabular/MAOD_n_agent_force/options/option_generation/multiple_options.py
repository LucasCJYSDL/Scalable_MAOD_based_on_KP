import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from options.graph.cover_time import AddEdge
import random
import time


def _decentralized_to_tuple(joint_state_list, A_list):
    agent_num = len(A_list)
    shape_list = []
    for agent_id in range(agent_num):
        shape_list.append(A_list[agent_id].shape[0])

    tuple_list = []
    for i in range(len(joint_state_list)):
        joint_state = joint_state_list[i]
        individual_state_list = []
        for agent_id in range(agent_num-1, 0, -1):
            left_idx = joint_state // shape_list[agent_id]
            right_idx = joint_state % shape_list[agent_id]
            individual_state_list.append(right_idx)
            joint_state = left_idx

        assert left_idx < shape_list[0]
        individual_state_list.append(left_idx) # for agent 0
        individual_state_list.reverse()
        tuple_list.append(tuple(individual_state_list))

    return tuple_list


def _kronecker_product(mdp, A_list, intToS, num_limit, normlized=True, with_degree=True, use_median=False):
    assert normlized

    agent_num = len(intToS)

    SToInt = [{} for _ in range(len(intToS))]
    for idx in range(len(intToS)):
        temp_dict = intToS[idx]
        for key, value in temp_dict.items():
            SToInt[idx][value] = key

    value_list = []
    vector_list = []
    diag_list = []
    for i in range(len(A_list)):
        diag_i = []
        for j in range(A_list[i].shape[0]):
            diag = A_list[i][j].sum()
            diag_i.append(diag)
        diag_list.append(diag_i)

        temp_Gnx = nx.to_networkx_graph(A_list[i])
        # print("The adjacency matrix of Agent {} is {}!".format(i, nx.to_numpy_matrix(temp_Gnx)))
        if normlized:
            temp_lap = nx.linalg.laplacianmatrix.normalized_laplacian_matrix(temp_Gnx).astype(float)
        else:
            temp_lap = nx.linalg.laplacianmatrix.laplacian_matrix(temp_Gnx).astype(float)
        # print("The laplacian matrix of Agent {} is {}!".format(i, temp_lap))
        value_i, vector_i = eigh(temp_lap.toarray())
        value_list.append(value_i)
        vector_list.append(vector_i)

        # print("The eigenvalue list of Agent {} is {}!".format(i, value_list))

    # get the degree list of the kronecker graph
    # degree_list = np.kron(diag_list[0], diag_list[1])
    # for idx in range(2, len(diag_list)):
    #     degree_list = np.kron(degree_list, diag_list[idx])
    if agent_num < 4:
        dim_0 = value_list[0].shape[0]
        dim_1 = value_list[1].shape[0]
        u_list = []
        for i in range(dim_0):
            for j in range(dim_1):
                u_ij = (value_list[0][i] + value_list[1][j] - value_list[0][i] * value_list[1][j])
                u_list.append(((i, j), u_ij))

        for idx in range(2, len(value_list)):
            v_list = []
            for i in range(len(u_list)):
                for j in range(value_list[idx].shape[0]):
                    v_ij = u_list[i][1] + value_list[idx][j] - u_list[i][1] * value_list[idx][j]
                    temp_list = list(u_list[i][0]) + [j]
                    v_list.append((tuple(temp_list), v_ij))
            u_list = v_list

        # u_list.sort(key=lambda x: x[1])
        # print("Sorted product eigenvalue: ", u_list[:1000])
        # 1: just the 2nd smallest
        assert abs(u_list[0][1] - 0.0) <= 1e-6
        # if abs(u_list[0][1] - u_list[1][1]) <= 1e-6:
        #     if u_list[0][0][0] < u_list[1][0][0]:
        #         ind_i, ind_j = u_list[0][0]
        #     else:
        #         ind_i, ind_j = u_list[1][0]
        # else:
        #     ind_i, ind_j = u_list[1][0]
        candidates_list = []
        for ele in u_list:
            if abs(ele[1] - 0.0) <= 1e-6:
                candidates_list.append(list(ele[0]))
        assert len(candidates_list) >= 2
        candidates_list.sort(key=lambda x:np.sum(x))
        print("Sorted Candidates List: ", candidates_list)
        idx_list = candidates_list[0]
    else:
        idx_list = [0, 0, 0, 0]

    init_set = {}
    vector_dict_list = [{} for _ in range(len(idx_list))]
    for idx in range(len(idx_list)):
        temp_vector = vector_list[idx][:, idx_list[idx]]
        for key, value in SToInt[idx].items():
            vector_dict_list[idx][key] = temp_vector[value]
    init_set['eigen'] = vector_dict_list

    min_init_set = init_set.copy()
    min_init_set['sign'] = '-'
    max_init_set = init_set.copy()
    max_init_set['sign'] = '+'

    v_ij = np.kron(vector_list[0][:, idx_list[0]], vector_list[1][:, idx_list[1]])
    for idx in range(2, len(idx_list)):
        v_ij = np.kron(v_ij, vector_list[idx][:, idx_list[idx]])
    v_ij = np.around(v_ij, decimals=8) # DO NOT comment this line!!!

    # if with_degree:
    #     v_ij = v_ij / np.sqrt(degree_list)

    max_v = np.max(v_ij)
    min_v = np.min(v_ij)

    if use_median:
        threshold = np.median(v_ij)
        min_init_set['threshold'] = threshold
        max_init_set['threshold'] = threshold
    else:
        min_init_set['threshold'] = max_v
        max_init_set['threshold'] = min_v

    max_list = np.argwhere(v_ij == max_v).flatten().tolist()
    min_list = np.argwhere(v_ij == min_v).flatten().tolist()

    random.shuffle(max_list)
    random.shuffle(min_list)
    # print("MAX_LIST: ", max_list)
    # print("MIN_LIST: ", min_list)

    # if num_limit > 0:
    #     if len(max_list) > num_limit:
    #         max_list = max_list[:num_limit]
    #     if len(min_list) > num_limit:
    #         min_list = min_list[:num_limit]

    init_list = _decentralized_to_tuple(max_list, A_list)
    term_list = _decentralized_to_tuple(min_list, A_list)
    # print("INIT_LIST: ", init_list)
    # print("TERM_LIST: ", term_list)

    # filter based on the env reward
    final_init_list = []
    final_term_list = []

    for init_s in init_list:
        if len(final_init_list) >= num_limit:
            break
        is_append = True
        for idx in range(len(intToS)):
            if not mdp.is_goal_state_single(intToS[idx][init_s[idx]]):
                is_append = False
                break
        if is_append:
            final_init_list.append(init_s)
    print("There are {} init states that are fully within the goal area!".format(len(final_init_list)))

    for init_s in init_list:
        if len(final_init_list) >= num_limit:
            break
        is_append = False
        for idx in range(len(intToS)):
            if mdp.is_goal_state_single(intToS[idx][init_s[idx]]):
                is_append = True
                break
        if is_append:
            final_init_list.append(init_s)
    print("There are {} init states that are fully or partly within the goal area!".format(len(final_init_list)))

    for init_s in init_list:
        if len(final_init_list) >= num_limit:
            break
        final_init_list.append(init_s)
    assert len(final_init_list) == num_limit

    for term_s in term_list:
        if len(final_term_list) >= num_limit:
            break
        is_append = True
        for idx in range(len(intToS)):
            if not mdp.is_goal_state_single(intToS[idx][term_s[idx]]):
                is_append = False
                break
        if is_append:
            final_term_list.append(term_s)
    print("There are {} term states that are fully within the goal area!".format(len(final_term_list)))

    for term_s in term_list:
        if len(final_term_list) >= num_limit:
            break
        is_append = False
        for idx in range(len(intToS)):
            if mdp.is_goal_state_single(intToS[idx][term_s[idx]]):
                is_append = True
                break
        if is_append:
            final_term_list.append(term_s)
    print("There are {} term states that are fully or partly within the goal area!".format(len(final_term_list)))

    for term_s in term_list:
        if len(final_term_list) >= num_limit:
            break
        final_term_list.append(term_s)
    assert len(final_term_list) == num_limit

    return final_init_list, final_term_list, min_init_set, max_init_set


def multi_options(mdp, G_list, intToS, generation_time, num_limit, normalized=True, with_degree=True, use_median=False):
    agent_num = len(G_list)
    option_list = []
    partition_list = []

    for i in range(agent_num):
        assert nx.is_connected(nx.to_networkx_graph(G_list[i])) # danger!!!

    cur_option_num = 0
    A_list = G_list.copy() # essential

    for _ in range(generation_time):
        # print("The algebraic connectivity of the joint state space is {}!".format(_get_algebraic_connectivity(A_list)))
        init_list, term_list, min_init_set, max_init_set = _kronecker_product(mdp, A_list, intToS, num_limit, normlized=normalized, with_degree=with_degree, use_median=use_median)
        cur_option_num += len(init_list) * len(term_list) * 2

        for i in range(len(init_list)):
            for j in range(len(term_list)):
                option_list.append([init_list[i], term_list[j]])
                partition_list.append([min_init_set.copy(), max_init_set.copy()])
                for k in range(agent_num):
                    A_list[k] = AddEdge(A_list[k], init_list[i][k], term_list[j][k]) # max->min and min->max
                    # print("The state transition graph of Agent {} is:".format(k))
                    # for row in A_list[k]:
                    #     print(row)

    return A_list, option_list, cur_option_num, partition_list


if __name__ == "__main__":
    A_2 = np.array([[0, 1, 0, 0],
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0]])

    # diag_entrys = []
    # for i in range(A_2.shape[0]):
    #     diag = A_2[i].sum()
    #     diag_entrys.append(diag)
    # print(diag_entrys)
    # diag_entrys = np.sort(diag_entrys)
    # print(diag_entrys)

    G_2 = nx.to_networkx_graph(A_2)
    print(G_2)
    # L_2 = nx.linalg.laplacian_matrix(G_2).astype(float)
    # print(L_2)
    L_2 = nx.linalg.laplacianmatrix.laplacian_matrix(G_2).astype(float)
    print(L_2)
    # L_2 = nx.linalg.laplacianmatrix.normalized_laplacian_matrix(G_2).astype(float)
    # print(L_2)
    #
    # evalues, evectors = eigsh(L_2, k=3, which='SA') # k must be smaller than N # check the multiply, compare the results with eigh
    # print("evalues: ", evalues)
    # print("evectors: ", evectors) # normalized
    values, vectors = eigh(L_2.toarray())
    print(values, vectors)
    kp = np.kron(vectors[:, 0], vectors[:, 1])
    print(kp)
    # kp = np.around(kp, decimals=6)
    # print(kp)
    # kp_m = np.max(kp)
    # print(kp_m)
    # kp_ml = np.argwhere(kp==np.max(kp)).flatten().tolist()
    # print(kp_ml)
    # #
    # kp_mil = np.argwhere(kp == np.min(kp)).flatten().tolist()
    # print(kp_mil)

    # alg_conn = nx.algebraic_connectivity(G_2) # adj matrix while not the lapa matrix
    # print(alg_conn)
    #
    # from options.graph.spectrum import ComputeFiedlerVector
    # f_vector = ComputeFiedlerVector(G_2)
    # print(f_vector)