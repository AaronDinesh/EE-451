import torch.nn as nn

########## MODEL 1 WORKS WITH GRID 20 ##########
class TinyYOLO(nn.Module):
    def __init__(self, num_classes=13):
        super(TinyYOLO, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 320x320

            nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 160x160

            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 80x80

            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 40x40

            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 20x20

            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.1),
        )

        # Final detection layer: [B, 20, 20, anchors * (5 + num_classes)]
        self.output = nn.Conv2d(512, 3 * (5 + num_classes), 1)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        # Reshape: (batch, anchors, 5 + num_classes, S, S)
        bs, _, h, w = x.shape
        return x.view(bs, 3, 5 + self.output.out_channels // 3 - 5, h, w)
