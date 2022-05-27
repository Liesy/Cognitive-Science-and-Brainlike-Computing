# reference https://zhuanlan.zhihu.com/p/410031664
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

seed = 42
torch.manual_seed(seed)
dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MyLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MyLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.hidden_state = None

        self.lstm_block = nn.LSTM(input_size=input_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def hidden_initialize(self, batch_size):
        num_layers = self.num_layers * 2 if self.lstm_block.bidirectional else self.num_layers
        h0 = torch.randn((num_layers, batch_size, self.hidden_size), dtype=dtype, device=device)
        c0 = torch.randn((num_layers, batch_size, self.hidden_size), dtype=dtype, device=device)
        return h0, c0

    def forward(self, x):
        self.hidden_state = self.hidden_initialize(x.shape[0])
        """
        LSTM的两个输出
        - output为LSTM所有时间步的隐层结果 (batch_size,seq_len,hidden_dim)
        - h为LSTM最后一个时间步的隐层结果，c为LSTM最后一个时间步的Cell状态，以元组形式产生
        """
        output, self.hidden_state = self.lstm_block(x, self.hidden_state)
        output = self.output_layer(output[:, -1:, :])
        pred = torch.sigmoid(output[:, -1:, :])  # (B,1,8)
        return pred
