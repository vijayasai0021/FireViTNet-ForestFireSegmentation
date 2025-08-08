import torch
import torch.nn as nn
import torch.nn.functional as F

class StripePooling(nn.Module):
    """Stripe Pooling for capturing spatial context"""
    def __init__(self, in_channels, pool_size=20):
        super().__init__()
        self.pool_size = pool_size
        
        # Horizontal and vertical pooling
        self.pool_h = nn.AdaptiveAvgPool2d((pool_size, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, pool_size))
        
        # Convolutions for processing pooled features
        self.conv_h = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (pool_size, 1), bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv_w = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (1, pool_size), bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Horizontal stripe pooling
        h_pool = self.pool_h(x)  # B x C x pool_size x 1
        h_conv = self.conv_h(h_pool)  # B x C x 1 x 1
        h_feat = F.interpolate(h_conv, size=(H, W), mode='bilinear', align_corners=False)
        
        # Vertical stripe pooling
        w_pool = self.pool_w(x)  # B x C x 1 x pool_size
        w_conv = self.conv_w(w_pool)  # B x C x 1 x 1
        w_feat = F.interpolate(w_conv, size=(H, W), mode='bilinear', align_corners=False)
        
        # Fusion
        fused = torch.cat([h_feat, w_feat], dim=1)
        out = self.fusion(fused)
        
        return out + x  # Residual connection
