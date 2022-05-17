import numpy as np
from typing import Tuple
from agents.spectral.learners import option_generator
from agents.single.SAC_agent import SACAgent

class Option(object):
    def __init__(self, args, sub_goal, threshold, sign, intra_policy, og_agent: option_generator.OptionGenerator):
        self.args = args
        self.sub_goal = sub_goal
        self.threshold = threshold
        self.sign = sign
        self.intra_policy = intra_policy
        self.og_agent = og_agent

    def is_init_true(self, state, agent_id):
        pass

    def is_term_true(self, state, agent_id):
        pass

    def act(self, state, agent_id):
        pass


class MultiAgentOption(Option):
    def __init__(self, args, sub_goal, threshold, sign, intra_policy: Tuple[SACAgent], og_agent):
        super().__init__(args, sub_goal, threshold, sign, intra_policy, og_agent)
        self.agent_num = len(sub_goal)

    def is_init_true(self, state, agent_id=None):
        return True
        # embedding = self.og_agent.get_multiple_embeddings(state)
        # if self.sign == '-':
        #     if embedding > self.threshold:
        #         return True
        #     else:
        #         return False
        # else:
        #     assert self.sign == '+'
        #     if embedding < self.threshold:
        #         return True
        #     else:
        #         return False

    def is_term_true(self, state, agent_id):
        obs = state[agent_id]
        return np.linalg.norm(np.array(obs)[: 2] - np.array(self.sub_goal[agent_id])) <= self.args.range_threshold

    def act(self, state, agent_id) -> np.ndarray:
        return self.intra_policy[agent_id].get_action(np.array(state[agent_id]))


class SingleAgentOption(Option):
    def __init__(self, args, sub_goal, threshold, sign, intra_policy: SACAgent, og_agent):
        super().__init__(args, sub_goal, threshold, sign, intra_policy, og_agent)

    def is_init_true(self, state, agent_id):
        return True
        # embedding = self.og_agent.get_single_embeddings(state, agent_id)
        # if self.sign == '-':
        #     if embedding > self.threshold:
        #         return True
        #     else:
        #         return False
        # else:
        #     assert self.sign == '+'
        #     if embedding < self.threshold:
        #         return True
        #     else:
        #         return False

    def is_term_true(self, state, agent_id):
        obs = state[agent_id]
        return np.linalg.norm(np.array(obs)[: 2] - np.array(self.sub_goal)) <= self.args.range_threshold

    def act(self, state, agent_id) -> np.ndarray:
        return self.intra_policy.get_action(np.array(state[agent_id]))
