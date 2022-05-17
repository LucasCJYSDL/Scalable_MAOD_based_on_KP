import torch.nn as nn
import torch.nn.functional as F


class MADDPGRNNAgent(nn.Module): # TODO: normalize inputs may help
    def __init__(self, input_shape, args):
        super(MADDPGRNNAgent, self).__init__()
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
            # initialize small to prevent saturation
            self.fc2.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

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
        q = self.out_fn(self.fc2(h))
        return q, h

