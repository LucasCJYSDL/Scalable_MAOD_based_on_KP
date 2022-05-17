import torch
import torch.nn as nn
import torch.nn.functional as F

class CentralRNNAgent(nn.Module):
    # input_shape = obs_shape + n_actions + n_agents
    def __init__(self, input_shape, args):
        super(CentralRNNAgent, self).__init__()
        self.args = args
        self.name = 'central_rnn'

        ## main part
        self.main_input = nn.Linear(input_shape, args.central_rnn_hidden_dim)
        # self.main_input = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.main_fc_1 = nn.Linear(args.central_rnn_hidden_dim, args.central_rnn_hidden_dim)
        self.main_fc_2 = nn.Linear(args.central_rnn_hidden_dim, args.central_rnn_hidden_dim)
        self.main_rnn = nn.GRUCell(args.central_rnn_hidden_dim, args.central_rnn_hidden_dim)
        self.main_fc_3 = nn.Linear(args.central_rnn_hidden_dim, args.central_rnn_hidden_dim)
        self.main_fc_4 = nn.Linear(args.central_rnn_hidden_dim, args.central_rnn_hidden_dim)
        self.main_output = nn.Linear(args.central_rnn_hidden_dim, args.n_actions * args.central_action_embed)

    def init_hidden(self):
        # make hidden states on the same device and with the same type
        return self.main_fc_1.weight.new(1, self.args.central_rnn_hidden_dim).zero_()

    def forward(self, obs, hidden_state):

        main_hidden = self.main_input(obs)
        # main_hidden = self.main_input(obs)
        main_skip_1 = main_hidden
        main_out_1 = F.relu(self.main_fc_1(main_hidden))
        main_out_1 = self.main_fc_2(main_out_1)
        main_out_1 += main_skip_1
        main_out_1 = F.relu(main_out_1)

        h_in = hidden_state.reshape(-1, self.args.central_rnn_hidden_dim)
        # print("7: ", h_in.size())
        h = self.main_rnn(main_out_1, h_in)

        main_skip_2 = h
        main_out_2 = F.relu(self.main_fc_3(h))
        main_out_2 = self.main_fc_4(main_out_2)
        main_out_2 += main_skip_2
        main_out_2 = F.relu(main_out_2)
        q = self.main_output(main_out_2)
        q = q.reshape(-1, self.args.n_actions, self.args.central_action_embed)

        return q, h
