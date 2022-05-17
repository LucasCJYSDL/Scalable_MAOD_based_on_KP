import numpy as np
import random
from simple_rl.planning.ValueIterationClass import ValueIteration


def GetAdjacencyMatrix(mdp, agent_id):
    # print('mdp type=', type(mdp))
    vi = ValueIteration(mdp, agent_id=agent_id) # TODO: the VI class does sampling which doesn't work for stochastic planning.
    # vi.run_vi() # time waste --> value function is not required
    vi.compute_matrix_from_trans_func()
    A, states = vi.compute_adjacency_matrix() # totally based on the transition function
    # A: numpy.array(N*N), states: list of GridWorldState
    for k in range(A.shape[0]):
        A[k][k] = 0

    intToS = dict()
    for i, s in enumerate(states):
        intToS[i] = s
    return A, intToS

def GetIncidenceMatrix(mdp, agent_id, n_traj=1, eps_len=10):
    '''
    Sample transitions and build amn incidence matrix.
    Returns: A: incidence matrix
             states: mapping from matrix index to state
    '''
    # TODO: What is the best way to represent the incidence matrix?
    # Required output: 

    pairs = [] # List of state transition pairs (s, s')
    hash_to_ind = dict() # Dictionary of state -> index
    ind_to_s = dict()

    actions = mdp.get_actions()
    cur_s = mdp.get_init_states()[agent_id]

    cur_h = hash(cur_s)
    hash_to_ind[cur_h] = 0
    ind_to_s[0] = cur_s

    n_states = 1

    # Sample transitions
    for i in range(n_traj):
        mdp.reset()
        cur_s = mdp.get_init_states()[agent_id]
        cur_h = hash(cur_s)
        
        for j in range(eps_len):
            # TODO: Sample trajectory based on a particular policy rather than a random walk?
            a = random.choice(actions)
            _, next_s, _ = mdp.execute_agent_action_single(a, agent_id=agent_id)
            # TODO: The GymMDP is not returning the correct observation???
            next_h = hash(next_s)

            if next_h in hash_to_ind.keys():
                next_i = hash_to_ind[next_h]
            else:
                # print('next_s=', next_s)
                next_i = n_states
                hash_to_ind[next_h] = next_i
                ind_to_s[next_i] = next_s
                n_states += 1

            p = (hash_to_ind[cur_h], next_i)
            pairs.append(p)

            cur_s = next_s
            cur_h = next_h

    # print('pairs=', pairs)
    
    matrix = np.zeros((n_states, n_states), dtype=int)

    print("The size of Agent {}'s state space is {}!".format(agent_id, n_states))

    for i in range(len(pairs)):
        if pairs[i][0] != pairs[i][1]:
            matrix[pairs[i][0], pairs[i][1]] = 1 # symmetric or not?
            matrix[pairs[i][1], pairs[i][0]] = 1 # TODO: Try both!

    mdp.reset()
    return matrix, ind_to_s

