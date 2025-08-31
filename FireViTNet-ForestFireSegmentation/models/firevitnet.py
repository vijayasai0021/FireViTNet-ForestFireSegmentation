import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# Assuming these files are in the same 'models' directory
from .cbam import CBAM
from .dense_aspp import DenseASPP
from .stripe_pooling import StripePooling

class FireViTNet(nn.Module):
    """
    FireViTNet model for forest fire segmentation, based on the paper's parallel architecture.
    This implementation uses a pre-trained MobileViT-S backbone from the 'timm' library.
    """
    def __init__(self, num_classes=1, input_size=(224, 224)):
        super(FireViTNet, self).__init__()
        self.input_size = input_size
        
        # --- Backbone ---
        # We use a reliable, pre-trained MobileViT (small version) from timm.
        # This avoids potential bugs in a custom implementation and gives us better starting weights.
        self.backbone = timm.create_model('mobilevit_s', pretrained=True, features_only=True)
        
        # Programmatically get the number of output channels from the backbone's final stage
        backbone_out_channels = self.backbone.feature_info.channels()[-1]  # For mobilevit_s, this is 640
        
        # --- Path 1: Multi-scale & Spatial Context (Top path in the diagram) ---
        self.aspp = DenseASPP(in_channels=backbone_out_channels, inter_channels=256, out_channels=256)
        self.stripe_pooling = StripePooling(in_channels=256)
        self.conv_path1 = nn.Conv2d(256, 256, kernel_size=1, bias=False)
        self.bn_path1 = nn.BatchNorm2d(256)

        # --- Path 2: Attention (Bottom path in the diagram) ---
        self.cbam = CBAM(gate_channels=backbone_out_channels)
        self.conv_path2 = nn.Conv2d(backbone_out_channels, 256, kernel_size=1, bias=False)
        self.bn_path2 = nn.BatchNorm2d(256)

        # --- Decoder ---
        # Takes the combined features and upsamples them to the final mask
        self.decoder_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False)
        self.decoder_bn1 = nn.BatchNorm2d(128)
        
        self.final_conv = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        # --- Feature extraction ---
        features = self.backbone(x)
        last_features = features[-1]  # Get the final, high-level feature map

        # --- Parallel Paths ---
        # Path 1
        path1 = self.aspp(last_features)
        path1 = self.stripe_pooling(path1)
        path1 = self.bn_path1(self.conv_path1(path1))
        
        # Path 2
        path2 = self.cbam(last_features)
        path2 = self.bn_path2(self.conv_path2(path2))

        # --- Combine Paths ---
        # The paper's diagram shows an addition operation
        combined_features = F.relu(path1 + path2)

        # --- Decode to segmentation map ---
        # Upsample by a factor of 4
        decoded = F.interpolate(combined_features, scale_factor=4, mode='bilinear', align_corners=True)
        decoded = F.relu(self.decoder_bn1(self.decoder_conv1(decoded)))
        
        # Final classification layer
        out = self.final_conv(decoded)
        
        # --- Upsample to original input size ---
        out = F.interpolate(out, size=self.input_size, mode='bilinear', align_corners=True)
        
        # Apply sigmoid to get a probability map (0 to 1) for the mask
        return torch.sigmoid(out)

# This block allows you to run this file directly to test the model's structure
if __name__ == '__main__':
    # Create a dummy input tensor to test the model
    # Batch size = 2, Channels = 3, Height = 224, Width = 224
    dummy_input = torch.randn(2, 3, 224, 224)
    
    # Instantiate the model
    model = FireViTNet(num_classes=1, input_size=(224, 224))
    
    # Pass the dummy input through the model
    output = model(dummy_input)
    
    # Print the output shape to verify it's correct
    print(f"Model instantiated successfully.")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Should be [2, 1, 224, 224]

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1_000_000:.2f} M")
