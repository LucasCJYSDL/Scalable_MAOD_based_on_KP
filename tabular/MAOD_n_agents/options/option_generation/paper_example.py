import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh


if __name__ == "__main__":
    A_1 = np.array([[0, 1],
                    [1, 0]])

    A_2 = np.array([[0, 1, 0, 0],
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0]])

    diag_entrys_1 = []
    for i in range(A_1.shape[0]):
        diag = A_1[i].sum()
        diag_entrys_1.append(diag)
    diag_entrys_2 = []
    for i in range(A_2.shape[0]):
        diag = A_2[i].sum()
        diag_entrys_2.append(diag)
    print(diag_entrys_1, '\n', diag_entrys_2)
    diag_entrys_1 = np.sort(diag_entrys_1)
    diag_entrys_2 = np.sort(diag_entrys_2)
    print(diag_entrys_1, '\n', diag_entrys_2)

    G_1 = nx.to_networkx_graph(A_1)
    G_2 = nx.to_networkx_graph(A_2)
    # L_1 = nx.linalg.laplacianmatrix.laplacian_matrix(G_1).astype(float)
    # L_2 = nx.linalg.laplacianmatrix.laplacian_matrix(G_2).astype(float)
    # print(L_1, '\n', L_2)
    L_1 = nx.linalg.laplacianmatrix.normalized_laplacian_matrix(G_1).astype(float)
    L_2 = nx.linalg.laplacianmatrix.normalized_laplacian_matrix(G_2).astype(float)
    print(L_1, '\n', L_2)

    values_1, vectors_1 = eigh(L_1.toarray())
    print(values_1, '\n', vectors_1)
    values_2, vectors_2 = eigh(L_2.toarray())
    print(values_2, '\n', vectors_2)

    joint_values = []
    for i in range(A_1.shape[0]):
        for j in range(A_2.shape[1]):
            u_ij = diag_entrys_1[i] * diag_entrys_2[j] * (values_1[i] + values_2[j] - values_1[i] * values_2[j])
            joint_values.append((u_ij, (i,j)))
    joint_values.sort(key=lambda x: x[0])
    print(joint_values)

    ind_i, ind_j = joint_values[2][1]
    kp = np.kron(vectors_1[:, ind_i], vectors_2[:, ind_j])
    print(kp)
    # # kp = np.around(kp, decimals=6)
    # # print(kp)
    # # kp_m = np.max(kp)
    # # print(kp_m)
    # # kp_ml = np.argwhere(kp==np.max(kp)).flatten().tolist()
    # # print(kp_ml)
    # # #
    # # kp_mil = np.argwhere(kp == np.min(kp)).flatten().tolist()
    # # print(kp_mil)
    #
    # alg_conn = nx.algebraic_connectivity(G_2) # adj matrix while not the lapa matrix
    # print(alg_conn)
    #
    # from options.graph.spectrum import ComputeFiedlerVector
    # f_vector = ComputeFiedlerVector(G_2)
    # print(f_vector)