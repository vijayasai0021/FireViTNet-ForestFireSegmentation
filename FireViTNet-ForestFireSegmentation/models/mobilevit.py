import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MV2Block(nn.Module):
    """MobileViT MV2 Block with depthwise separable convolutions"""
    def __init__(self, in_channels, out_channels, stride=1, expansion=6):
        super().__init__()
        hidden_dim = int(in_channels * expansion)
        
        self.conv = nn.Sequential(
            # Pointwise convolution
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            
            # Depthwise convolution
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            
            # Pointwise convolution
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        self.use_res_connect = stride == 1 and in_channels == out_channels
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileViTBlock(nn.Module):
    """Core MobileViT block combining convolutions with transformers"""
    def __init__(self, in_channels, transformer_dim, ffn_dim, n_heads=4, patch_size=2, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.transformer_dim = transformer_dim
        
        # Local representation layers
        self.local_rep1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.local_rep2 = nn.Conv2d(in_channels, transformer_dim, 1)
        
        # Transformer layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Global representation layer
        self.global_rep = nn.Conv2d(transformer_dim, in_channels, 1)
        
        # Layer normalization
        self.norm = nn.LayerNorm(transformer_dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Local representation
        local_rep = F.relu(self.local_rep1(x))
        local_rep = self.local_rep2(local_rep)
        
        # Unfold into patches
        patches = F.unfold(local_rep, kernel_size=self.patch_size, stride=self.patch_size)
        patches = rearrange(patches, 'b (c p1 p2) n -> b n (c p1 p2)', 
                          p1=self.patch_size, p2=self.patch_size)
        
        # Apply transformer
        patches = self.norm(patches)
        transformed = self.transformer(patches)
        
        # Fold back to feature map
        patches = rearrange(transformed, 'b n (c p1 p2) -> b (c p1 p2) n',
                          p1=self.patch_size, p2=self.patch_size,
                          c=self.transformer_dim)
        
        output_h = H // self.patch_size
        output_w = W // self.patch_size
        folded = F.fold(patches, output_size=(output_h, output_w),
                       kernel_size=1, stride=1)
        
        # Interpolate back to original size
        folded = F.interpolate(folded, size=(H, W), mode='bilinear', align_corners=False)
        
        # Global representation
        global_rep = self.global_rep(folded)
        
        return global_rep + x  # Residual connection

class MobileViTBackbone(nn.Module):
    """Complete MobileViT backbone for feature extraction"""
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1
        self.stage1 = nn.Sequential(
            MV2Block(16, 32, 1),
            MV2Block(32, 32, 2),
            MV2Block(32, 32, 1),
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            MV2Block(32, 64, 2),
            MobileViTBlock(64, 96, 384, n_heads=4, patch_size=2),
            MV2Block(64, 64, 1),
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            MV2Block(64, 128, 2),
            MobileViTBlock(128, 144, 576, n_heads=4, patch_size=2),
            MV2Block(128, 128, 1),
        )
        
    def forward(self, x):
        features = []
        
        x = self.stem(x)  # /2
        features.append(x)
        
        x = self.stage1(x)  # /4
        features.append(x)
        
        x = self.stage2(x)  # /8
        features.append(x)
        
        x = self.stage3(x)  # /16
        features.append(x)
        
        return features
