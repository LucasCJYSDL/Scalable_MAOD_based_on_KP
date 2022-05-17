import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.multiple.utils.rl_utils import AddBias


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args): # TODO: initialization of the network weights may help
        super(RNNAgent, self).__init__()
        self.args = args

        if self.args.is_discrete:
            output_dim = args.n_actions
        else:
            output_dim = args.action_shape

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, output_dim)

        if not self.args.is_discrete:
            self.logstd = AddBias(torch.zeros(output_dim))

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)

        if self.args.is_discrete:
            return q, h
        else:
            zeros = torch.zeros(q.size())
            if self.args.use_cuda:
                zeros = zeros.cuda()
            action_logstd = self.logstd(zeros)
            return q, action_logstd.exp(), h

