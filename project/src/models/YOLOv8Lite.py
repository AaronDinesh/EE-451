import torch.nn as nn
from src.modules.C2F import C2F
from src.modules.SPPF import SPPF

class YOLOv8Lite(nn.Module):
    def __init__(self, num_classes=13, anchors=3):
        super().__init__()
        self.anchors = anchors
        self.num_classes = num_classes

        def cbs(in_ch, out_ch, k=3, s=1, p=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, s, p),
                nn.BatchNorm2d(out_ch),
                nn.SiLU()
        )

        self.stem = nn.Sequential(
            cbs(3, 64, 3, 2, 1),
            C2F(64, 128),
            cbs(128, 128, 3, 2, 1),
            C2F(128, 256),
            cbs(256, 256, 3, 2, 1),
            C2F(256, 512),
            cbs(512, 512, 3, 2, 1),
            C2F(512, 512),
        )

        self.neck = nn.Sequential(
            SPPF(512, 512),
            cbs(512, 512, 3, 1, 1)
        )

        self.head = nn.Conv2d(512, anchors * (5 + num_classes), 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.neck(x)
        x = self.head(x)
        bs, _, h, w = x.shape
        return x.view(bs, self.anchors, 5 + self.num_classes, h, w)
