import torch as th
from torch.distributions import Categorical
REGISTRY = {}

class SoftPoliciesSelector():

    def __init__(self, args):
        self.args = args

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        m = Categorical(agent_inputs)
        picked_actions = m.sample().long()
        return picked_actions


REGISTRY["soft_policies"] = SoftPoliciesSelector

if __name__ == '__main__':
    test = SoftPoliciesSelector(None)
    inputs = th.tensor([[[0.1, 0.4, 0.5], [0.4, 0.2, 0.4]], [[0.1, 0.3, 0.6], [0.1, 0.7, 0.2]]])
    print(inputs.shape)
    actions = test.select_action(inputs, None, None)
    print(actions, actions.shape)