import logging

import torch
import torch.nn.functional as F

import numpy as np

from layers.layers import *

class MoodRecog(torch.nn.Module):
    def __init__(self, config):
        super(MoodRecog, self).__init__()

        self.config = config

        # --------------- mel_enc ---------------- #
        self.first_net = Conv1d1x1(
            in_channels=self.config['mel_enc']['in_channels'],
            out_channels=self.config['mel_enc']['out_channels'],
            bias=self.config['mel_enc']['bias']
        )
        if config['mel_enc']['reset']:
            self.first_net.reset_parameters()
        self.second_net = torch.nn.ModuleList()
        for i in range(self.config['mel_enc']['resnet_n_layer']):
            if self.config['mel_enc']['batchnorm']:
                self.second_net.append(torch.nn.BatchNorm1d(num_features=self.config['mel_enc']['out_channels']))
            self.second_net.append(torch.nn.MaxPool1d(self.config['mel_enc']['maxpool_size']))
            self.second_net.append(torch.nn.ReLU(inplace=True))
            conv_ = Conv1d(
                in_channels=self.config['mel_enc']['out_channels'],
                out_channels=self.config['mel_enc']['out_channels'],
                kernel_size=self.config['mel_enc']['kernel_size'],
                padding=(self.config['mel_enc']['kernel_size'] - 1) // 2,
                bias=self.config['mel_enc']['bias']
            )
            if config['mel_enc']['reset']:
                conv_.reset_parameters()
            self.second_net.append(conv_)
        self.second_net.append(torch.nn.ReLU(inplace=True))

        # --------------- lstm ---------------- #
        self.lstm = torch.nn.LSTM(
            input_size=self.config['lstm']['input_size'],
            hidden_size=self.config['lstm']['hidden_size'],
            num_layers=self.config['lstm']['num_layers'],
            bias=self.config['lstm']['bias'],
            batch_first=self.config['lstm']['batch_first'],
            bidirectional=self.config['lstm']['bidirectional']
        )

        # --------------- linear ---------------- #
        self.linear = torch.nn.ModuleList()
        self.linear.append(torch.nn.Linear(
                in_features=self.config['linear']['in_features'],
                out_features=self.config['linear']['hidden_features'],
                bias=self.config['linear']['bias']
            ))
        if self.config['linear']['batchnorm']:
            self.linear.append(torch.nn.BatchNorm1d(num_features=self.config['linear']['hidden_features']))
        self.linear.append(torch.nn.ReLU(inplace=True))
        self.linear.append(torch.nn.Linear(
                in_features=self.config['linear']['hidden_features'],
                out_features=self.config['linear']['out_features'],
                bias=self.config['linear']['bias']
            ))
        if self.config['linear']['batchnorm']:
            self.linear.append(torch.nn.BatchNorm1d(num_features=self.config['linear']['out_features']))
        self.linear.append(torch.nn.Softmax(dim=1))
        # self.linear = torch.nn.Linear(
        #     in_features=self.config['linear']['in_features'],
        #     out_features=self.config['linear']['out_features'],
        #     bias=self.config['linear']['bias']
        # )

    def forward(self, x):
        # --------------- mel_enc ---------------- #
        # x: [B, C, L], batch, channels, length
        z = self.first_net(x)
        # res = z
        for f in self.second_net:
            z = f(z)
        # z = z + res

        # --------------- lstm ---------------- #
        # z: [B, C, L]
        z = z.permute(0, 2, 1)  # [B, C, L] -> [B, L, C] for lstm input
        z, (hn, cn) = self.lstm(z)
        # hn, cn: [D * num_layers, B, C], D = 2 if bidirectional else 1

        # --------------- linear ---------------- #
        hn = hn.permute(1, 0, 2)    # [D * num_layers, B, C] -> [B, D * num_layers, C]
        out = torch.flatten(hn, start_dim=1)     # [B, D * num_layers, C] -> [B, D * num_layers * C]
        # out = self.linear(out)
        for f in self.linear:
            out = f(out)
        # out = F.sigmoid(out)
        # NOTE: the sigmoid is integrated into the BCEWithLogits() of torch

        return out
