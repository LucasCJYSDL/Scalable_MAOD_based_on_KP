import numpy as np
import networkx as nx
from options.graph.cover_time import AddEdge
from options.graph.spectrum import ComputeFiedlerVector, ComputeConnectivity
import time

def _spectral_partitioning(A, v, degree_list, use_median):
    vol_V = np.sum(degree_list).astype(np.float64)
    state_num = len(v)

    joint_dict_list = []
    for idx in range(state_num):
        joint_dict_list.append({'id': idx, 'v': v[idx], 'd': degree_list[idx]})
    joint_dict_list.sort(key=lambda x: x['v'])

    last_E = 0.0
    vol_S = 0.0
    phi_list = []
    min_phi = float('inf')
    min_idx = -1
    if use_median:
        min_idx = state_num // 2
    else:
        for idx in range(state_num - 1):
            vol_S += joint_dict_list[idx]['d']
            if vol_S > vol_V / 2: # 1
                break             # 1
            curr_d = joint_dict_list[idx]['d']
            inner_edges = 0
            for last_idx in range(idx):
                if A[joint_dict_list[last_idx]['id']][joint_dict_list[idx]['id']] > 0:
                    inner_edges += 1
            curr_E = curr_d + last_E - 2 * inner_edges
            curr_phi = float(curr_E) / vol_S  # 1
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
        if idx <= min_idx:
            min_S_list.append(joint_dict_list[idx]['id'])
        else:
            max_S_list.append(joint_dict_list[idx]['id'])

    return min_S_list, max_S_list


def single_options(G_list, generation_time, with_degree, use_median):
    agent_num = len(G_list)
    option_list = [[] for _ in range(agent_num)]
    partition_list = [[] for _ in range(agent_num)]

    for i in range(agent_num):
        assert nx.is_connected(nx.to_networkx_graph(G_list[i])) # danger!!!

    cur_option_num = 0
    A_list = G_list.copy()

    for _ in range(generation_time):
        # build the options separately
        for i in range(agent_num):
            A_i = A_list[i]
            state_num_i = len(A_i)
            degree_list_i = []
            for idx in range(state_num_i):
                degree_list_i.append(np.sum(A_i[idx]))
            degree_list_i = np.array(degree_list_i).astype(np.float64)
            vector_i = ComputeFiedlerVector(nx.to_networkx_graph(A_i))  # A is Adjacency Matrix while not the Laplacian Matrix
            #conn_i = ComputeConnectivity(A_i)
            vector_i = np.around(vector_i, decimals=8)

            if with_degree:
                vector_i = vector_i / np.sqrt(degree_list_i)
            print("Spectral Partitioning begins ......")
            start = time.time()
            min_S_list, max_S_list = _spectral_partitioning(A_i, vector_i, degree_list_i, use_median)
            print("Time Cost: ", time.time() - start)

            max_list = np.argwhere(vector_i == np.max(vector_i)).flatten().tolist()
            min_list = np.argwhere(vector_i == np.min(vector_i)).flatten().tolist()
            # print("FiedlerVector: ", vector_i)
            print("MAX_LIST: ", max_list)
            print("MIN_LIST: ", min_list)

            cur_option_num += len(max_list) * len(min_list) * 2
            for init_point in max_list:
                for term_point in min_list:
                    option_list[i].append((init_point, term_point))
                    partition_list[i].append((min_S_list, max_S_list))
                    A_list[i] = AddEdge(A_list[i], init_point, term_point) #deep copy or shallow copy
                    # assert A_list[i][init_point][term_point] == 1
            # print("The state transition graph of Agent {} is:".format(i))
            # for row in A_list[i]:
            #     print(row)

    return A_list, option_list, cur_option_num, partition_list


if __name__ == '__main__':
    A_2 = np.array([[0, 1, 0, 0],
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0]])

    # v = ComputeFiedlerVector(nx.to_networkx_graph(A_2))  # A is Adjacency Matrix while not the Laplacian Matrix
    # lmd = ComputeConnectivity(A_2)
    #
    # print(lmd, '\n', v)
    A_list = []
    A_list.append(A_2.copy())
    A_list.append(A_2.copy())
    print(A_list)

    A_list[0] = AddEdge(A_list[0], 0, 2)
    print(A_list)
    A_list[0] = AddEdge(A_list[0], 1, 1)
    print(A_list)
