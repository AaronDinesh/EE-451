import torch

class DFL(torch.nn.Module):
    def __init__(self, in_channels=16):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(in_channels, dtype=torch.float)
        self.conv1.weight.data = torch.nn.Parameter(x.view(1, in_channels, 1, 1))
        self.in_channels = in_channels

    def forward(self, x):
        b, _, a = x.shape
        return self.conv1(x.view(b, 4, self.in_channels, a).transpose(2, 1).softmax(1)).view(b, 4, a)