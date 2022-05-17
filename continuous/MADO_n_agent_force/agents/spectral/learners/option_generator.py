from typing import List
import numpy as np

from agents.spectral.learners.laprepr import LapReprLearner
from agents.spectral.configs import get_generator_args
from agents.spectral.utils.timer_tools import Timer

class OptionGenerator(object):
    def __init__(self, args, eigenvalue_list, laprepr_list: List[LapReprLearner]):
        self.args = get_generator_args(args)
        self.laprepr_list = laprepr_list
        self.eigenvalue_list = eigenvalue_list
        self.agent_num = len(self.eigenvalue_list)
        assert self.agent_num >= 2

    def get_multi_options(self):
        print("Start to collect the multi-agent options!")
        timer = Timer()
        # get the eigenvalue list
        dim = self.args.d
        u_list = []
        for i in range(dim):
            for j in range(dim):
                u_ij = (self.eigenvalue_list[0][i] + self.eigenvalue_list[1][j] - self.eigenvalue_list[0][i] * self.eigenvalue_list[1][j])
                u_list.append(((i, j), u_ij))

        for idx in range(2, self.agent_num):
            v_list = []
            for i in range(len(u_list)):
                for j in range(dim):
                    v_ij = u_list[i][1] + self.eigenvalue_list[idx][j] - u_list[i][1] * self.eigenvalue_list[idx][j]
                    temp_list = list(u_list[i][0]) + [j]
                    v_list.append((tuple(temp_list), v_ij))
            u_list = v_list

        u_list.sort(key=lambda x: x[1])
        print("The sorted eigenvalue list: ", u_list)
        self.fiedler_list = u_list[1][0] # TODO: the second smallest, more tricks can be applied here

        # find the sub-goal states, danger
        optimum_list = []
        for idx in range(self.agent_num):
            print("Calculating the optimum for Agent {}.".format(idx))
            optimum_list.append(self.laprepr_list[idx].get_embedding_optimum(self.args.og_n_samples, self.args.og_interval,
                                                                             dim=self.fiedler_list[idx], with_degree=self.args.with_degree))

        joint_optimum_list = []
        for i in range(2):
            for j in range(2):
                joint_optimum_list.append(((optimum_list[0][i][0], optimum_list[1][j][0]), optimum_list[0][i][1] * optimum_list[1][j][1]))

        for idx in range(2, self.agent_num):
            temp_optimum_list = []
            for i in range(len(joint_optimum_list)):
                for j in range(2):
                    temp_optimum = joint_optimum_list[i][1] * optimum_list[idx][j][1]
                    temp_joint_state = list(joint_optimum_list[i][0]) + [optimum_list[idx][j][0]]
                    temp_optimum_list.append((tuple(temp_joint_state), temp_optimum))
            joint_optimum_list = temp_optimum_list

        joint_optimum_list.sort(key=lambda x: x[1])
        print("Joint optimum list is {}!!!".format(joint_optimum_list))
        self.min_joint_state = joint_optimum_list[0][0] # tuple[np.ndarray]
        self.max_joint_state = joint_optimum_list[-1][0]
        self.min_value = joint_optimum_list[0][1]
        self.max_value = joint_optimum_list[-1][1]

        self.threshold = self.min_value + (self.max_value - self.min_value) * self.args.threshold

        sub_goal_list = [self.min_joint_state, self.max_joint_state]

        for joint_optimum in joint_optimum_list:
            sub_goals = joint_optimum[0]
            value = joint_optimum[1]
            assert len(sub_goals) == self.agent_num
            is_put = True
            for idx in range(self.agent_num):
                if not self.laprepr_list[idx].env.is_goal_area(sub_goals[idx]):
                    is_put = False
                    break
            if is_put:
                print("!!!Perfect!!!")
                if value <= self.threshold:
                    sub_goal_list[0] = sub_goals
                else:
                    sub_goal_list[1] = sub_goals


        print("The collection of multi-agent options is completed, with time cost {:.4g}s.".format(timer.time_cost()))
        print("!!! The multiple-agent sub_goal list is {}!!!".format(sub_goal_list))

        return sub_goal_list, self.threshold

    def get_multiple_embeddings(self, state):
        assert len(state) == self.agent_num
        embedding_list = []
        for idx in range(self.agent_num):
            embedding_list.append(self.laprepr_list[idx].get_embedding(data_input=state[idx], dim=self.fiedler_list[idx], with_degree=self.args.with_degree))

        return np.prod(embedding_list)

    def get_single_options(self):
        # find the sub-goal states, danger
        print("Start to collect the single-agent options!")
        timer = Timer()
        state_list = []
        threshold_list = []
        for idx in range(self.agent_num):
            print("Calculating the optimum for Agent {}.".format(idx))
            temp_list = self.laprepr_list[idx].get_embedding_optimum(self.args.og_n_samples, self.args.og_interval,
                                                                             dim=1, with_degree=self.args.with_degree)
            state_list.append((temp_list[0][0], temp_list[1][0]))
            threshold_list.append(temp_list[0][1] + (temp_list[1][1] - temp_list[0][1]) * self.args.threshold)

        print("The collection of single-agent options is completed, with time cost {:.4g}s.".format(timer.time_cost()))
        print("!!! The single-agent sub_goal list is {}!!!".format(state_list))

        return state_list, threshold_list

    def get_single_embeddings(self, state, agent_id):
        return self.laprepr_list[agent_id].get_embedding(data_input=state[agent_id], dim=1, with_degree=self.args.with_degree)


