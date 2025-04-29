import torch

class C2F(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n=2):
        super().__init__()
        hidden_channels = out_channels // 2
        self.conv1 = torch.nn.Conv2d(in_channels, hidden_channels, 1, 1, 0)
        self.conv2 = torch.nn.Conv2d(in_channels, hidden_channels, 1, 1, 0)
        
        #This is trying to emulate the bottleneck layer in YOLO? BUt I dont think it is exactly correct...
        self.blocks = torch.nn.Sequential(
            *[torch.nn.Sequential(
                torch.nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
                torch.nn.BatchNorm2d(hidden_channels),
                torch.nn.SiLU()
            ) for _ in range(n)]
        )
        self.out_conv = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_channels * (n + 1), out_channels, 1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.SiLU()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        outputs = [x1]
        for block in self.blocks:
            x1 = block(x1)
            outputs.append(x1)
        return self.out_conv(torch.cat(outputs, dim=1))
