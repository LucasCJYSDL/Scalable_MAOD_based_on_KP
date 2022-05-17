import torch as th
from torch import Tensor
from torch.autograd import Variable
from agents.multiple.utils.rl_utils import OUNoise, gumbel_softmax, onehot_from_logits
from agents.multiple.modules.agents import REGISTRY as agent_REGISTRY


# This multi-agent controller shares parameters between agents
class MADDPGMAC:
    def __init__(self, scheme, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = None
        if not args.is_discrete:
            self.exploration = OUNoise(args.action_shape)
        else:
            self.exploration = self.args.init_noise  # epsilon for eps-greedy, seems not be used

        self.hidden_states = None

    def reset_noise(self):
        if not self.args.is_discrete:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.args.is_discrete:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def select_actions(self, ep_batch, t_ep, t_env=0, test_mode=False):
        # Only select actions for the selected batch elements in bs
        agent_outputs = self.forward(ep_batch, t_ep)
        if self.args.is_discrete:
            if not test_mode:
                chosen_actions = gumbel_softmax(agent_outputs, hard=True).argmax(dim=-1) # (bs, n_agent)
            else:
                chosen_actions = onehot_from_logits(agent_outputs).argmax(dim=-1) # (bs, n_agent)
        else:
            if not test_mode:
                agent_outputs += Variable(Tensor(self.exploration.noise(agent_outputs)), requires_grad=False).to(self.args.device) # (bs, n_agent, action_shape)
            chosen_actions = agent_outputs.clamp(-1.0, 1.0) # the limit of the continuous actions
        return chosen_actions

    def target_actions(self, ep_batch, t_ep):
        agent_outputs = self.forward(ep_batch, t_ep)
        if self.args.is_discrete:
            return onehot_from_logits(agent_outputs) # (bs, n_agent, n_actions)
        else:
            return agent_outputs.clamp(-1.0, 1.0) # (bs, n_agent, action_shape)

    def forward(self, ep_batch, t):
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        agent_outs = agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
        if self.args.is_discrete:
            avail_actions = ep_batch["avail_actions"][:, t]
            agent_outs[avail_actions==0] = -1e10
        return agent_outs

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def init_hidden_one_agent(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av

        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
