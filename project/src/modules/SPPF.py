import torch

class SPPF(torch.nn.Module):
    def __init__(self, in_channels, out_channels, pool_size=5):
        super().__init__()
        self.cv1 = torch.nn.Conv2d(in_channels, out_channels // 2, 1, 1)
        self.cv2 = torch.nn.Conv2d(out_channels // 2 * 4, out_channels, 1, 1)
        self.pool = torch.nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=pool_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))
