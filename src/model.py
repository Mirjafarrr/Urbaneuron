import torch
import torch.nn as nn
import torchvision.models as models


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ResNeXt50Encoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2 if pretrained else None
        resnext = models.resnext50_32x4d(weights=weights)

        self.stem = nn.Sequential(resnext.conv1, resnext.bn1, resnext.relu)
        self.maxpool = resnext.maxpool
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

    def forward(self, x):
        x00 = self.stem(x)
        x10 = self.maxpool(x00)
        x10 = self.layer1(x10)
        x20 = self.layer2(x10)
        x30 = self.layer3(x20)
        x40 = self.layer4(x30)
        return x00, x10, x20, x30, x40


class UNetPlusPlusDecoderResNeXt(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        d0, d1, d2, d3 = 64, 128, 256, 512

        self.conv_01 = DoubleConv(in_channels=64 + 256, out_channels=d0)
        self.conv_02 = DoubleConv(in_channels=64 + d0 + d1, out_channels=d0)
        self.conv_03 = DoubleConv(in_channels=64 + d0 * 2 + d1, out_channels=d0)
        self.conv_04 = DoubleConv(in_channels=64 + d0 * 3 + d1, out_channels=d0)
        self.conv_11 = DoubleConv(in_channels=256 + 512, out_channels=d1)
        self.conv_12 = DoubleConv(in_channels=256 + d1 + d2, out_channels=d1)
        self.conv_13 = DoubleConv(in_channels=256 + d1 * 2 + d2, out_channels=d1)
        self.conv_21 = DoubleConv(in_channels=512 + 1024, out_channels=d2)
        self.conv_22 = DoubleConv(in_channels=512 + d2 + d3, out_channels=d2)
        self.conv_31 = DoubleConv(in_channels=1024 + 2048, out_channels=d3)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.final_upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.final_conv = nn.Conv2d(in_channels=d0, out_channels=num_classes, kernel_size=1)

    def forward(self, x00, x10, x20, x30, x40):
        x01 = self.conv_01(torch.cat([x00, self.upsample(x10)], dim=1))
        x11 = self.conv_11(torch.cat([x10, self.upsample(x20)], dim=1))
        x21 = self.conv_21(torch.cat([x20, self.upsample(x30)], dim=1))
        x31 = self.conv_31(torch.cat([x30, self.upsample(x40)], dim=1))
        x02 = self.conv_02(torch.cat([x00, x01, self.upsample(x11)], dim=1))
        x12 = self.conv_12(torch.cat([x10, x11, self.upsample(x21)], dim=1))
        x22 = self.conv_22(torch.cat([x20, x21, self.upsample(x31)], dim=1))
        x03 = self.conv_03(torch.cat([x00, x01, x02, self.upsample(x12)], dim=1))
        x13 = self.conv_13(torch.cat([x10, x11, x12, self.upsample(x22)], dim=1))
        x04 = self.conv_04(torch.cat([x00, x01, x02, x03, self.upsample(x13)], dim=1))
        output = self.final_upsample(x04)
        output = self.final_conv(output)
        return output


class UNetPlusPlusResNeXt50(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.encoder = ResNeXt50Encoder(pretrained=pretrained)
        self.decoder = UNetPlusPlusDecoderResNeXt(num_classes=num_classes)

    def forward(self, x):
        encoder_features = self.encoder(x)
        output = self.decoder(*encoder_features)
        return output