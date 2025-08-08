import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPPConv(nn.Sequential):
    """ASPP convolution with atrous convolution"""
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super().__init__(*modules)

class ASPPPooling(nn.Sequential):
    """ASPP pooling layer"""
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class DenseASPP(nn.Module):
    """Dense Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, atrous_rates=[6, 12, 18], out_channels=256):
        super().__init__()
        
        modules = []
        
        # 1x1 conv
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Atrous convolutions with dense connections
        prev_channels = in_channels
        for i, rate in enumerate(atrous_rates):
            modules.append(ASPPConv(prev_channels, out_channels, rate))
            prev_channels += out_channels  # Dense connection
        
        # Global average pooling
        modules.append(ASPPPooling(in_channels, out_channels))
        
        self.convs = nn.ModuleList(modules)
        
        # Final projection
        total_channels = out_channels * (len(atrous_rates) + 2)  # +2 for 1x1 and pooling
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )
        
    def forward(self, x):
        results = []
        
        # 1x1 conv
        results.append(self.convs[0](x))
        
        # Dense atrous convolutions
        input_feat = x
        for i in range(1, len(self.convs) - 1):
            feat = self.convs[i](input_feat)
            results.append(feat)
            input_feat = torch.cat([input_feat, feat], dim=1)
        
        # Global pooling
        results.append(self.convs[-1](x))
        
        # Concatenate all features
        out = torch.cat(results, dim=1)
        
        # Final projection
        return self.project(out)
