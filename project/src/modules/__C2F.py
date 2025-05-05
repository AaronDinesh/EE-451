import torch
from .__ConvLayer import ConvLayer
from .__BottleneckLayer import BottleneckLayer

class C2F(torch.nn.Module):
    def __init__(self, shortcut, in_channels, out_channels, n_iters):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, 2*out_channels, 1, 1, 0) #The diagram had it as out_channels and then had the in_channels for the next layer divided by 2.
        self.conv2 = ConvLayer((n_iters+2) * out_channels, out_channels, 1, 1, 0)
        self.bottleneck_list = torch.nn.ModuleList(BottleneckLayer(shortcut=shortcut, in_channels=out_channels) for _ in range(n_iters))

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottleneck_list)
        return self.conv2(torch.cat(y, 1))
