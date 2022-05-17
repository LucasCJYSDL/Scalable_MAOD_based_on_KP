import gym

from simulation import mujoco_maze
from agents.single.SAC_agent import SACAgent
from agents.spectral.utils.timer_tools import Timer
from agents.spectral.learners.option_wrapper import MultiAgentOption, SingleAgentOption
from agents.spectral.learners.option_generator import OptionGenerator

class OptionAgent(object):
    def __init__(self, args, env_id_list, og_agent: OptionGenerator):
        self.args = args
        self.agent_num = len(env_id_list)
        self.og_agent = og_agent
        self.timer = Timer()

        self.env_list = []
        for idx in range(self.agent_num):
            temp_env_id = env_id_list[idx][:-1] + '1' # the corresponding environment with reward shaping
            self.env_list.append(gym.make(temp_env_id))

    def get_multi_option_agents(self, sub_goal_list, threshold):
        # collect the intra-option list
        intra_policy_list = []
        for i in range(2): # min or max
            temp_list = []
            for idx in range(self.agent_num):
                print("Start to train the intra-option policy toward the {} subgoal of Agent {}!".format('MIN' if i==0 else 'MAX', idx))
                self.timer.reset()
                temp_agent = SACAgent(self.env_list[idx], new_goal_area={'goal': sub_goal_list[i][idx], 'threshold': self.args.range_threshold})
                temp_agent.learn()
                temp_list.append(temp_agent)
                print("Training ends with time cost: {:.4g}s.".format(self.timer.time_cost()))
            intra_policy_list.append(tuple(temp_list))

        multi_option_list = []
        for i in range(2): # min or max
            sign = '-' if i == 0 else '+'
            multi_option_list.append(MultiAgentOption(self.args, sub_goal_list[i], threshold, sign, intra_policy_list[i], self.og_agent))

        return multi_option_list


    def get_single_option_agents(self, sub_goal_list, threshold_list):
        # collect the intra-option list
        intra_policy_list = []
        for idx in range(self.agent_num):
            temp_list = []
            for i in range(2):
                print("Start to train the intra-option policy toward the {} subgoal of Agent {}!".format('MIN' if i == 0 else 'MAX', idx))
                self.timer.reset()
                temp_agent = SACAgent(self.env_list[idx], new_goal_area={'goal': sub_goal_list[idx][i], 'threshold': self.args.range_threshold})
                temp_agent.learn()
                temp_list.append(temp_agent)
                print("Training ends with time cost: {:.4g}s.".format(self.timer.time_cost()))
            intra_policy_list.append(tuple(temp_list))

        single_option_list = []
        for idx in range(self.agent_num):
            temp_option_list = []
            for i in range(2):
                sign = '-' if i == 0 else '+'
                temp_option_list.append(SingleAgentOption(self.args, sub_goal_list[idx][i], threshold_list[idx], sign, intra_policy_list[idx][i], self.og_agent))
            single_option_list.append(tuple(temp_option_list))

        return single_option_list
