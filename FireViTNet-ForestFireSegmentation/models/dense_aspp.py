import torch
import torch.nn as nn
import torch.nn.functional as F

class _DenseASPPConv(nn.Sequential):
    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate, drop_rate=0.1):
        super(_DenseASPPConv, self).__init__()
        self.add_module(
            "conv1",
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
        ),
        self.add_module(
            "bn1",
            nn.BatchNorm2d(inter_channels),
        ),
        self.add_module(
            "relu1",
            nn.ReLU(inplace=True),
        ),
        self.add_module(
            "conv2",
            nn.Conv2d(inter_channels, out_channels, kernel_size=3, dilation=atrous_rate, padding=atrous_rate, bias=False),
        ),
        self.add_module(
            "bn2",
            nn.BatchNorm2d(out_channels),
        ),
        self.add_module(
            "relu2",
            nn.ReLU(inplace=True),
        )
        self.drop_rate = drop_rate

    def forward(self, x):
        features = super(_DenseASPPConv, self).forward(x)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.training)
        return features

class DenseASPP(nn.Module):
    """
    Implementation of the Dense Atrous Spatial Pyramid Pooling module.
    """
    def __init__(self, in_channels, inter_channels=256, out_channels=256):
        super(DenseASPP, self).__init__()
        
        # The paper's diagram shows 5 dilated convolutions plus a pooling layer.
        # Let's use the dilation rates mentioned in similar architectures.
        atrous_rates = [3, 6, 12, 18, 24]

        # Each new convolution takes the original input + all previous outputs
        self.aspp_conv1 = _DenseASPPConv(in_channels, inter_channels, out_channels, atrous_rates[0])
        self.aspp_conv2 = _DenseASPPConv(in_channels + out_channels * 1, inter_channels, out_channels, atrous_rates[1])
        self.aspp_conv3 = _DenseASPPConv(in_channels + out_channels * 2, inter_channels, out_channels, atrous_rates[2])
        self.aspp_conv4 = _DenseASPPConv(in_channels + out_channels * 3, inter_channels, out_channels, atrous_rates[3])
        self.aspp_conv5 = _DenseASPPConv(in_channels + out_channels * 4, inter_channels, out_channels, atrous_rates[4])

        # Final convolution layer to combine all the feature maps
        self.project = nn.Sequential(
            nn.Conv2d(in_channels + 5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        aspp1 = self.aspp_conv1(x)
        x2 = torch.cat([aspp1, x], dim=1)

        aspp2 = self.aspp_conv2(x2)
        x3 = torch.cat([aspp2, x2], dim=1)

        aspp3 = self.aspp_conv3(x3)
        x4 = torch.cat([aspp3, x3], dim=1)
        
        aspp4 = self.aspp_conv4(x4)
        x5 = torch.cat([aspp4, x4], dim=1)
        
        aspp5 = self.aspp_conv5(x5)

        return self.project(torch.cat([aspp1, aspp2, aspp3, aspp4, aspp5, x], dim=1))
