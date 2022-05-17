from agents.multiple.MARL_agent import MARLAgent
from agents.spectral.spectral_agent import SpectralAgent

class SingleHierarchicalAgent(object):

    def __init__(self, args, option_list):
        self.args = args
        self.option_list = option_list #[(agent_0_min: np.ndarray, agent_0_max), (agent_1_min, agent_1_max), ...]
        self.agent_num = len(self.option_list)
        assert self.agent_num == self.args.agent_num

    def update_option_list(self):
        option_num = len(self.option_list[0])
        while option_num < self.args.option_num:
            new_spectral_agent = SpectralAgent(env_id=self.args.env_id, seed=self.args.seed, agent_num=self.agent_num)
            new_option_list = new_spectral_agent.get_option_list(mode='single', option_list=self.option_list)
            temp_option_list = []
            for idx in range(self.agent_num):
                temp_option_list.append(tuple(list(self.option_list[idx]) + list(new_option_list[idx])))
            self.option_list = temp_option_list

    def setup_agents(self):
        option_num = len(self.option_list[0])
        self.low_level_agent = MARLAgent(env_dict={"task": self.args.env_id, "type": "continuous"}, alg=self.args.low_level_alg, seed=self.args.seed)
        self.high_level_agent = MARLAgent(env_dict={"task": self.args.env_id, "type": "hierarchical", "n_actions": option_num + 1},
                                          alg=self.args.high_level_alg, seed=self.args.seed)
        self.max_training_steps = self.low_level_agent.get_max_training_steps()

    def learn(self):
        self.low_level_agent.learn(training_steps=500000) # catch up with the intra-option policys

        step_interval = 50000
        max_episode = (self.max_training_steps // step_interval) + 1
        for i in range(max_episode):
            print("Hierarchical Training {}/{}......".format(i, max_episode))
            self.high_level_agent.learn(training_steps=step_interval, low_level_mac=self.low_level_agent.run_agent.mac, option_list=self.option_list)
            # if i % 2 == 1:
            #     self.low_level_agent.learn(training_steps=step_interval)

        print("All the training is completed!!!!!!")