#!/bin/python

import numpy as np

#############################
# This util file should be independent to MDP module.
#############################

def neighbor(graph, n):
    '''
    Args:
        G (numpy 2d array): Adjacency matrix
        n (integer): index of the node
    Returns:
        (list of integers): neighbor nodes
    Summary:
        Given a graph adjacency matrix and a node, return a list of its neighbor nodes.
    '''
    assert(graph.ndim == 2)
    # print('graph=', graph)
    array = (np.array(graph))[n]
    # print('array=', array)
    # print('graph=', type(graph), '\n', graph)
    l = []
    for i in range(len(array)):
        
        if array[i] == 1:
            l.append(i)
    return l

def AddEdge(G, vi, vj):
    augGraph = G.copy()
    # print('augGraph', augGraph)
    if vi != vj: # no self-loop
        augGraph[vi, vj] = 1
        augGraph[vj, vi] = 1
    return augGraph

