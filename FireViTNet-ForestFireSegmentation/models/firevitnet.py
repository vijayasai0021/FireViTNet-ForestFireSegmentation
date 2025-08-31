import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming these imports point to your custom module files
from .mobilevit import MobileViTBackbone 
from .cbam import CBAM
from .dense_aspp import DenseASPP
from .stripe_pooling import StripePooling

class FireViTNet(nn.Module):
    """FireViTNet model with the parallel architecture from the paper."""
    
    # Using num_classes=1 for binary segmentation is more standard
    def __init__(self, num_classes=1, in_channels=3):
        super().__init__()
        
        # --- Backbone ---
        self.backbone = MobileViTBackbone(in_channels)
        # IMPORTANT: You must know the output channels of your backbone.
        # Let's assume your backbone's final feature map has 128 channels.
        backbone_out_channels = 128 
        
        # --- Path 1: Multi-scale & Spatial Context ---
        self.aspp = DenseASPP(backbone_out_channels, atrous_rates=[6, 12, 18], out_channels=256)
        self.stripe_pooling = StripePooling(256)
        # 1x1 Conv to match channel dimensions for combining
        self.conv_path1 = nn.Conv2d(256, 256, kernel_size=1, bias=False)

        # --- Path 2: Attention ---
        self.cbam = CBAM(backbone_out_channels)
        # 1x1 Conv to match channel dimensions for combining
        self.conv_path2 = nn.Conv2d(backbone_out_channels, 256, kernel_size=1, bias=False)

        # --- Decoder ---
        # The decoder now takes the combined features as input
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Dropout is a good addition for regularization
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
        
        # --- Feature extraction ---
        features = self.backbone(x)
        last_features = features[-1] # Use final features
        
        # --- Parallel Paths ---
        # Path 1
        path1 = self.aspp(last_features)
        path1 = self.stripe_pooling(path1)
        path1 = self.conv_path1(path1)

        # Path 2
        path2 = self.cbam(last_features)
        path2 = self.conv_path2(path2)
        
        # --- Combine Paths ---
        # The diagram shows an addition operation
        combined_features = path1 + path2
        
        # --- Decode to segmentation map ---
        decoded_map = self.decoder(combined_features)
        
        # --- Upsample to original size ---
        out = F.interpolate(decoded_map, size=input_size, mode='bilinear', align_corners=False)
        
        # For binary segmentation (num_classes=1), a sigmoid is typically applied
        # either here or by the loss function (BCEWithLogitsLoss).
        # If you keep num_classes=2, you would apply a softmax.
        return torch.sigmoid(out) if self.decoder[-1].out_channels == 1 else out

    # Your helper functions get_model_size() and count_parameters() are great and can remain as they are.
