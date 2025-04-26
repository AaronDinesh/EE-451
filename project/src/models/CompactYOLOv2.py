import torch.nn as nn

########## MODEL 2 WORKS WITH GRID 40 ##########
class CompactYOLOv2(nn.Module):
    def __init__(self, num_classes=13, anchors=3):
        super(CompactYOLOv2, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.MaxPool2d(2, 2),  # 320

            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.MaxPool2d(2, 2),  # 160

            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.SiLU(),
            nn.MaxPool2d(2, 2),  # 80

            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.SiLU(),
            nn.MaxPool2d(2, 2),  # 40

            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.SiLU(),
            nn.MaxPool2d(2, 2),  # 20
        )

        self.head = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.SiLU(),
            nn.Conv2d(256, anchors * (5 + num_classes), 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        bs, _, h, w = x.shape
        return x.view(bs, self.anchors, 5 + self.num_classes, h, w)
