import torch
from .ConvLayer import ConvLayer

class SPPF(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        c_ = in_channels // 2
        self.conv1 = ConvLayer(in_channels, c_, 1, 1, 0)
        self.conv2 = ConvLayer(c_*4, out_channels, 1, 1, 0)

        self.maxpool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    
    def forward(self, x):
        y = [self.conv1(x)]
        y.extend(self.maxpool(y[-1]) for _ in range(3))
        return self.conv2(torch.cat(y, 1))