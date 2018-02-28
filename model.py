# -*- coding: utf-8 -*-
"""
A pure implementation of the AlphaGo's Network part, the network's structure looks like LeNet
The original verion is written by:
@author: Junxiao Song
@github: https://github.com/junxiaosong/AlphaZero_Gomoku/blob/master/policy_value_net_pytorch.py

It is modified to simplify the code by:
@author: Yuan Liu
@github: https://github.com/cmusjtuliuyuan/AlphaGoZero/blob/master/model.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*board_width*board_height, board_width*board_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.softmax(self.act_fc1(x_act), dim=1)
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height, net_params=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Net(board_width, board_height).cuda()
        else:
            self.policy_value_net = Net(board_width, board_height)
        self.optimizer = optim.SGD(self.policy_value_net.parameters(),
                                   weight_decay=self.l2_const,
                                   momentum=0.9,
                                   lr=0.001)

        if net_params:
            self.policy_value_net.load_state_dict(net_params)

    def policy_value_fn(self, board):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        state_batch = board.current_state().reshape(-1, 4, self.board_width, self.board_height)
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            act_probs, value = self.policy_value_net(state_batch)
            return act_probs.data.cpu().numpy()[0], value.data.cpu().numpy()[0]
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            act_probs, value = self.policy_value_net(state_batch)
            return act_probs.data.numpy()[0], value.data.numpy()[0]

    def train_batch(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2 (Note: the L2 penalty is incorporated in optimizer)
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * torch.log(act_probs), 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(torch.sum(act_probs * torch.log(act_probs), 1))
        return loss.data[0], entropy.data[0]

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params
