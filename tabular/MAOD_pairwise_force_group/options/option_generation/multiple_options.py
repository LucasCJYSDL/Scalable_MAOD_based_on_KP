import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from options.graph.cover_time import AddEdge
import random
import time

def _is_connected(last_idx, curr_idx, dim_1, A_list):
    tuple_list = _decentralized_to_tuple([last_idx, curr_idx], dim_1)
    last_tuple = tuple_list[0]
    curr_tuple = tuple_list[1]

    assert len(last_tuple) == len(curr_tuple) == len(A_list)
    for idx in range(len(A_list)):
        if int(A_list[idx][last_tuple[idx]][curr_tuple[idx]]) == 0:
            return False
    return True


# spectral partitioning -- v: normalized Fielder vector
def _spectral_partitioning(A_list, degree_list, v, intToS, dim_1, use_median):
    vol_V = np.sum(degree_list).astype(np.float64)
    state_num = len(v)

    joint_dict_list = []
    for idx in range(state_num):
        joint_dict_list.append({'id': idx, 'v': v[idx], 'd': degree_list[idx]})
    joint_dict_list.sort(key=lambda x:x['v'])
    # new_v = []
    # for idx in range(state_num):
    #     new_v.append(joint_dict_list[idx]['v'])

    last_E = 0.0
    vol_S = 0.0
    phi_list = []
    min_phi = float('inf')
    min_idx = -1
    if use_median:
        print("Use the Median directly!!!")
        min_idx = state_num // 2
    else:
        for idx in range(state_num-1):
            # vol_S = np.sum(new_v[:idx+1]).astype(np.float64)
            vol_S += joint_dict_list[idx]['d']
            if vol_S > vol_V / 2: # 1
                break             # 1
            curr_d = joint_dict_list[idx]['d']
            inner_edges = 0
            for last_idx in range(idx):
                if _is_connected(joint_dict_list[last_idx]['id'], joint_dict_list[idx]['id'], dim_1, A_list):
                    inner_edges += 1
            curr_E = curr_d + last_E - 2 * inner_edges
            curr_phi = float(curr_E) / vol_S # 1
            # curr_phi = float(curr_E) / min(vol_S, vol_V-vol_S) # 2

            phi_list.append(curr_phi)
            last_E = curr_E

            if curr_phi < min_phi:
                min_phi = curr_phi
                min_idx = idx

    assert min_idx > 0
    min_S_list = []
    max_S_list = []
    for idx in range(state_num):
        temp_tuple = _decentralized_to_tuple([joint_dict_list[idx]['id']], dim_1)[0]
        # state_list = []
        # for t_id in range(len(temp_tuple)):
        #     state_list.append(intToS[t_id][temp_tuple[t_id]])
        if idx <= min_idx:
            min_S_list.append(list(temp_tuple))
        else:
            max_S_list.append(list(temp_tuple))

    return min_S_list, max_S_list



def _decentralized_to_tuple(joint_state_list, dim_1):
    tuple_list = []
    for i in range(len(joint_state_list)):
        joint_state = joint_state_list[i]
        tuple_0 = joint_state // dim_1
        tuple_1 = joint_state % dim_1
        tuple_list.append((tuple_0, tuple_1))

    return tuple_list

def _kronecker_product(mdp, A_list, intToS, id_list, num_limit, normlized=True, with_degree=True, use_median=False):
    assert len(A_list) == 2, "For now, we only have the theorem for 2 agents......"
    assert normlized

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

    dim_0 = value_list[0].shape[0]
    dim_1 = value_list[1].shape[0]

    # get the degree list of the kronecker graph
    degree_list = np.kron(diag_list[0], diag_list[1]).astype(np.float64)

    u_list = []
    for i in range(dim_0):
        for j in range(dim_1):
            if normlized:
                u_ij = (value_list[0][i] + value_list[1][j] - value_list[0][i] * value_list[1][j])
            else:
                u_ij = value_list[0][i] * diag_list[1][j] + diag_list[0][i] * value_list[1][j] - value_list[0][i] * value_list[1][j]
            u_list.append(((i, j), u_ij))

    u_list.sort(key=lambda x: x[1])
    print("Sorted product eigenvalue: ", u_list[:1000])
    # 1: just the 2nd smallest
    assert abs(u_list[0][1] - 0.0) <= 1e-6
    if abs(u_list[0][1] - u_list[1][1]) <= 1e-6:
        if u_list[0][0][0] < u_list[1][0][0]:
            ind_i, ind_j = u_list[0][0]
        else:
            ind_i, ind_j = u_list[1][0]
    else:
        ind_i, ind_j = u_list[1][0]
    #2: the first one element that is not zero
    # ind_i = ind_j = -1
    # assert u_list[0][1] == 0.0
    # for ele in u_list:
    #     if ele[1] > 0:
    #         ind_i, ind_j = ele[0]
    #         break
    # assert ind_i >= 0 and ind_j >= 0
    # take the multiplicity of the 2nd eigenvalue into consideration, so we can collect more than one (i, j) pairs

    v_ij = np.kron(vector_list[0][:, ind_i], vector_list[1][:, ind_j])
    v_ij = np.around(v_ij, decimals=8) # DO NOT comment this line!!!

    if with_degree:
        v_ij = v_ij / np.sqrt(degree_list)
    print("Spectral Partitioning begins ......")
    start = time.time()
    min_S_list, max_S_list = _spectral_partitioning(A_list, degree_list, v_ij, intToS, dim_1, use_median)
    print("Time Cost: ", time.time() - start)

    max_list = np.argwhere(v_ij == np.max(v_ij)).flatten().tolist()
    min_list = np.argwhere(v_ij == np.min(v_ij)).flatten().tolist()
    random.shuffle(max_list)
    random.shuffle(min_list)
    print("MAX_LIST: ", max_list)
    print("MIN_LIST: ", min_list)

    # if num_limit > 0:
    #     if len(max_list) > num_limit:
    #         max_list = max_list[:num_limit]
    #     if len(min_list) > num_limit:
    #         min_list = min_list[:num_limit]

    init_list = _decentralized_to_tuple(max_list, dim_1)
    term_list = _decentralized_to_tuple(min_list, dim_1)
    print("INIT_LIST: ", init_list)
    print("TERM_LIST: ", term_list)

    # filter based on the env reward
    final_init_list = []
    final_term_list = []

    final_init_list = []
    final_term_list = []

    for init_s in init_list:
        if len(final_init_list) >= num_limit:
            break
        if mdp.is_goal_state_single(intToS[0][init_s[0]], agent_id=id_list[0]) and mdp.is_goal_state_single(intToS[1][init_s[1]],
                                                                                                            agent_id=id_list[1]):
            final_init_list.append(init_s)
    print("There are {} init states that are fully within the goal area!".format(len(final_init_list)))

    for init_s in init_list:
        if len(final_init_list) >= num_limit:
            break
        if mdp.is_goal_state_single(intToS[0][init_s[0]], agent_id=id_list[0]) or mdp.is_goal_state_single(intToS[1][init_s[1]], agent_id=id_list[1]):
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
        if mdp.is_goal_state_single(intToS[0][term_s[0]], agent_id=id_list[0]) and mdp.is_goal_state_single(intToS[1][term_s[1]],
                                                                                                            agent_id=id_list[1]):
            final_term_list.append(term_s)
    print("There are {} term states that are fully within the goal area!".format(len(final_term_list)))

    for term_s in term_list:
        if len(final_term_list) >= num_limit:
            break
        if mdp.is_goal_state_single(intToS[0][term_s[0]], agent_id=id_list[0]) or mdp.is_goal_state_single(intToS[1][term_s[1]], agent_id=id_list[1]):
            final_term_list.append(term_s)
    print("There are {} term states that are fully or partly within the goal area!".format(len(final_term_list)))

    for term_s in term_list:
        if len(final_term_list) >= num_limit:
            break
        final_term_list.append(term_s)
    assert len(final_term_list) == num_limit

    return final_init_list, final_term_list, min_S_list, max_S_list


def multi_options(mdp, G_list, intToS, id_list, generation_time, num_limit, normalized=True, with_degree=True, use_median=False):
    agent_num = len(G_list)
    option_list = []
    partition_list = []

    for i in range(agent_num):
        assert nx.is_connected(nx.to_networkx_graph(G_list[i])) # danger!!!

    cur_option_num = 0
    A_list = G_list.copy() # essential

    for _ in range(generation_time):
        # print("The algebraic connectivity of the joint state space is {}!".format(_get_algebraic_connectivity(A_list)))
        init_list, term_list, min_S_list, max_S_list = _kronecker_product(mdp, A_list, intToS, id_list, num_limit, normlized=normalized, with_degree=with_degree, use_median=use_median)
        cur_option_num += len(init_list) * len(term_list) * 2

        for i in range(len(init_list)):
            for j in range(len(term_list)):
                option_list.append([init_list[i], term_list[j]])
                partition_list.append([min_S_list, max_S_list])
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