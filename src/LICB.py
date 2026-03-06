import argparse
import datetime
import os

import lightning as L
import pandas as pd
import torch
import torch.nn as nn

from src.gltcn import TemporalCnn


class Tcn_Local(nn.Module):

    def __init__(self, num_outputs, kernel_size=3, dropout=0.2):  # k>=3
        """
        TCN, the current paper gives a TCN structure that supports well the case of one number per moment, i.e., the sequence structure.
        For a one-dimensional structure where each moment is a vector, it is barely possible to split the vector into several input channels at that moment.
        For the case where each moment is a matrix or a higher dimensional image, it is not so good.

        :param num_outputs: int, the number of output channels
        :param input_length: int, the length of the sliding window input sequence
        :param kernel_size: int, the size of the convolution kernel
        :param dropout: float, drop_out ratio
        """
        super(Tcn_Local, self).__init__()
        layers = []
        num_levels = 3
        out_channels = num_outputs
        for i in range(num_levels):
            layers += [TemporalCnn(out_channels, out_channels, kernel_size, stride=1, dilation=1,
                                   padding=(kernel_size - 1),
                                   dropout=dropout)]  # Adding padding to the convolved tensor to achieve causal convolution by slicing the tensor

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        The structure of input x is different from RNN, which generally has size (Batch, seq_len, channels) or (seq_len, Batch, channels).
        Here the seq_len is put after channels, and the data of all time steps are put together and used as the input size of Conv1d to realize the operation of convolution across time steps.
        Very clever design.

        :param x: size of (Batch, out_channel, seq_len)
        :return: size of (Batch, out_channel, seq_len)
        """
        return self.network(x)



class IICB(L.LightningModule):
    def __init__(self, in_features, drop=0.):
        super().__init__()
        self.conv1 = Tcn_Local(num_outputs=in_features, kernel_size=5, dropout=0.2)
        self.conv2 = Tcn_Local(num_outputs=in_features, kernel_size=4, dropout=0.2)
        self.conv3 = Tcn_Local(num_outputs=in_features, kernel_size=3, dropout=0.2)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2
        x = self.conv3(out1 + out2)

        return x
