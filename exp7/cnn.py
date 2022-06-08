import numpy as np
import torch
import torch.nn as nn

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, keep_size=False):
        super(ConvBlock, self).__init__()
        padding = (keep_size - 1) // 2 if keep_size else padding
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class MyCNN(nn.Module):
    def __init__(self, in_channels):
        super(MyCNN, self).__init__()
        self.in_channels = in_channels

        # (1,28,28) -> (6,12,12)
        self.conv_block_1 = ConvBlock(in_channels=in_channels, out_channels=6, kernel_size=5)
        # (6,12,12) -> (12,4,4)
        self.conv_block_2 = ConvBlock(in_channels=6, out_channels=12, kernel_size=5)

        self.output_layer = nn.Linear(in_features=12 * 4 * 4, out_features=10)

    def forward(self, img):
        B, C, H, W = img.shape
        assert C == self.in_channels
        f_map_1 = self.conv_block_1(img)
        f_map_2 = self.conv_block_2(f_map_1)
        output = self.output_layer(f_map_2.contiguous().view(B, -1))
        return output
