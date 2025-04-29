import torch

#helper padding function for convLayer
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, dilation=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=autopad(kernel_size, padding, dilation), groups=groups, dilation=dilation, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.silu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))