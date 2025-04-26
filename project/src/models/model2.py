import torch
import torch.nn as nn
from src.modules.ConvLayer import ConvLayer
from src.modules.BottleneckLayer import BottleneckLayer
from src.modules.C2F import C2F
from src.modules.SPPF import SPPF
from src.modules.DFL import DFL
from src.modules.Detect import Detect, make_anchors

class YOLOv8(nn.Module):
    """YOLOv8 object detection model"""
    def __init__(self, nc=80, variant='s'):
        super().__init__()
        # Variant determines the model's size
        assert variant in ['n', 's', 'm', 'l', 'x']
        
        # Scaling parameters based on variant
        scale_dict = {
            'n': [0.33, 0.25, 1024],  # depth, width, max_channels
            's': [0.33, 0.50, 1024],
            'm': [0.67, 0.75, 768],
            'l': [1.00, 1.00, 512],
            'x': [1.00, 1.25, 512]
        }
        
        depth_scale, width_scale, max_channels = scale_dict[variant]
        
        # Apply width scaling
        def width_scaled(x):
            return max(int(x * width_scale), 1)
        
        # Create channels and layer structures
        channels = [width_scaled(64), width_scaled(128), width_scaled(256), width_scaled(512), width_scaled(1024)]
        
        # Apply depth scaling (number of repeats)
        def depth_scaled(x):
            return max(round(x * depth_scale), 1)
        
        # Backbone
        self.backbone = nn.ModuleList()
        # P1/2 - First convolution
        self.backbone.append(ConvLayer(3, channels[0], 3, 2))  # 0
        # P2/4 - Second convolution
        self.backbone.append(ConvLayer(channels[0], channels[1], 3, 2))  # 1
        # P2/4 - C2f block
        self.backbone.append(C2F(True, channels[1], channels[1], depth_scaled(3)))  # 2
        # P3/8 - Third convolution
        self.backbone.append(ConvLayer(channels[1], channels[2], 3, 2))  # 3
        # P3/8 - C2f block
        self.backbone.append(C2F(True, channels[2], channels[2], depth_scaled(6)))  # 4
        # P4/16 - Fourth convolution
        self.backbone.append(ConvLayer(channels[2], channels[3], 3, 2))  # 5
        # P4/16 - C2f block
        self.backbone.append(C2F(True, channels[3], channels[3], depth_scaled(6)))  # 6
        # P5/32 - Fifth convolution
        self.backbone.append(ConvLayer(channels[3], channels[4], 3, 2))  # 7
        # P5/32 - C2f block
        self.backbone.append(C2F(True, channels[4], channels[4], depth_scaled(3)))  # 8
        # P5/32 - SPPF block
        self.backbone.append(SPPF(channels[4], channels[4], 5))  # 9
        
        # Neck/Head
        self.head = nn.ModuleList()
        # P5/32 -> P4/16 upsampling
        self.head.append(nn.Upsample(scale_factor=2, mode='nearest'))  # 10
        # Concatenate with P4/16 backbone
        self.head.append(C2F(False, channels[4] + channels[3], channels[3], depth_scaled(3)))  # 11
        
        # P4/16 -> P3/8 upsampling
        self.head.append(nn.Upsample(scale_factor=2, mode='nearest'))  # 12
        # Concatenate with P3/8 backbone
        self.head.append(C2F(False, channels[3] + channels[2], channels[2], depth_scaled(3)))  # 13
        
        # P3/8 -> P4/16 downsampling
        self.head.append(ConvLayer(channels[2], channels[2], 3, 2))  # 14
        # Concatenate with previous P4/16 feature
        self.head.append(C2F(False, channels[2] + channels[3], channels[3], depth_scaled(3)))  # 15
        
        # P4/16 -> P5/32 downsampling
        self.head.append(ConvLayer(channels[3], channels[3], 3, 2))  # 16
        # Concatenate with P5/32 backbone
        self.head.append(C2F(False, channels[3] + channels[4], channels[4], depth_scaled(3)))  # 17
        
        # Detection head
        self.detect = Detect(nc, [channels[2], channels[3], channels[4]])  # Channels for P3, P4, P5
        
        # Initialize strides and anchors
        self.stride = torch.tensor([8, 16, 32])  # strides for P3, P4, P5
        
    def forward(self, x):
        outputs = []  # outputs to store feature maps
        connections = {}  # to store feature maps for skip connections
        
        # Process backbone
        for i, module in enumerate(self.backbone):
            x = module(x)
            
            # Store feature maps needed for skip connections
            if i in [4, 6, 9]:  # P3/8, P4/16, P5/32 outputs
                connections[i] = x
            
            if i == 9:  # End of backbone
                break
        
        # Process neck and head
        # P5/32 path
        fpn_p5 = connections[9]  # P5 feature from SPPF
        
        # Upsample P5 and concat with P4 backbone
        fpn_p5_up = self.head[0](fpn_p5)  # Upsample
        p4_in = torch.cat([fpn_p5_up, connections[6]], 1)  # Concat with P4 backbone
        fpn_p4 = self.head[1](p4_in)  # Process P4
        
        # Upsample P4 and concat with P3 backbone
        fpn_p4_up = self.head[2](fpn_p4)  # Upsample
        p3_in = torch.cat([fpn_p4_up, connections[4]], 1)  # Concat with P3 backbone
        fpn_p3 = self.head[3](p3_in)  # Process P3
        outputs.append(fpn_p3)  # P3 output for detection
        
        # PAN: Downsample P3 and concat with P4
        pan_p3_down = self.head[4](fpn_p3)  # Downsample
        p4_in = torch.cat([pan_p3_down, fpn_p4], 1)  # Concat with previous P4
        pan_p4 = self.head[5](p4_in)  # Process P4
        outputs.append(pan_p4)  # P4 output for detection
        
        # PAN: Downsample P4 and concat with P5
        pan_p4_down = self.head[6](pan_p4)  # Downsample
        p5_in = torch.cat([pan_p4_down, fpn_p5], 1)  # Concat with previous P5
        pan_p5 = self.head[7](p5_in)  # Process P5
        outputs.append(pan_p5)  # P5 output for detection
        
        # Detection head
        return self.detect(outputs)

def yolov8_s(pretrained=False, nc=80):
    """Create YOLOv8 Small model"""
    model = YOLOv8(nc=nc, variant='s')
    if pretrained:
        # This would load pretrained weights
        pass
    return model

def yolov8_n(pretrained=False, nc=80):
    """Create YOLOv8 Nano model"""
    model = YOLOv8(nc=nc, variant='n')
    if pretrained:
        # This would load pretrained weights
        pass
    return model

def yolov8_m(pretrained=False, nc=80):
    """Create YOLOv8 Medium model"""
    model = YOLOv8(nc=nc, variant='m')
    if pretrained:
        # This would load pretrained weights
        pass
    return model

def yolov8_l(pretrained=False, nc=80):
    """Create YOLOv8 Large model"""
    model = YOLOv8(nc=nc, variant='l')
    if pretrained:
        # This would load pretrained weights
        pass
    return model

def yolov8_x(pretrained=False, nc=80):
    """Create YOLOv8 XLarge model"""
    model = YOLOv8(nc=nc, variant='x')
    if pretrained:
        # This would load pretrained weights
        pass
    return model

# Example usage
if __name__ == "__main__":
    # Create YOLOv8 small model
    model = yolov8_s(nc=80)
    
    # Test with a dummy input
    x = torch.randn(1, 3, 640, 640)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model structure: {model}")