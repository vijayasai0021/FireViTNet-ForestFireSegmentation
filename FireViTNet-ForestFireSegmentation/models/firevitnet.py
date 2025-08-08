import torch
import torch.nn as nn
import torch.nn.functional as F

from .mobilevit import MobileViTBackbone
from .cbam import CBAM
from .dense_aspp import DenseASPP
from .stripe_pooling import StripePooling

class FireViTNet(nn.Module):
    """Complete FireViTNet model for forest fire segmentation"""
    def __init__(self, num_classes=2, in_channels=3):
        super().__init__()
        
        # Backbone
        self.backbone = MobileViTBackbone(in_channels)
        
        # Attention module
        self.cbam = CBAM(128)  # Applied to final backbone features
        
        # Dense ASPP
        self.aspp = DenseASPP(128, atrous_rates=[6, 12, 18], out_channels=256)
        
        # Stripe pooling
        self.stripe_pooling = StripePooling(256)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(64, num_classes, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        input_size = x.shape[-2:]
        
        # Feature extraction
        features = self.backbone(x)
        x = features[-1]  # Use final features (1/16 resolution)
        
        # Apply attention
        x = self.cbam(x)
        
        # Multi-scale context
        x = self.aspp(x)
        
        # Spatial context
        x = self.stripe_pooling(x)
        
        # Decode to segmentation map
        x = self.decoder(x)
        
        # Upsample to original size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        return x
    
    def get_model_size(self):
        """Calculate model size in MB"""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
