import torch
from .__ConvLayer import ConvLayer

class BottleneckLayer(torch.nn.Module):
    def __init__(self, shortcut, in_channels):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, in_channels, 3, 1, 1)
        self.conv2 = ConvLayer(in_channels, in_channels, 3, 1, 1)
        self.shortcut = shortcut
    
    def forward(self, x):
        if self.shortcut:
            return x + self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x))

