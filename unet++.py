import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetPP(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, dropout=0.5):
        super(UNetPP, self).__init__()

        # Encoder blocks
        self.x0_0 = ConvBlock(in_channels, 32)
        self.x1_0 = ConvBlock(32, 64)
        self.x2_0 = ConvBlock(64, 128)
        self.x3_0 = ConvBlock(128, 256)
        self.x4_0 = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)

        # Decoder (nested skip connections)
        self.up3_1 = nn.Conv2d(512, 256, kernel_size=2, padding=1)
        self.x3_1 = ConvBlock(512, 256)

        self.up2_1 = nn.Conv2d(256, 128, kernel_size=2, padding=1)
        self.x2_1 = ConvBlock(256, 128)
        self.up2_2 = nn.Conv2d(256, 128, kernel_size=2, padding=1)
        self.x2_2 = ConvBlock(128 * 2, 128)

        self.up1_1 = nn.Conv2d(128, 64, kernel_size=2, padding=1)
        self.x1_1 = ConvBlock(128, 64)
        self.up1_2 = nn.Conv2d(128, 64, kernel_size=2, padding=1)
        self.x1_2 = ConvBlock(64 * 2, 64)
        self.up1_3 = nn.Conv2d(128, 64, kernel_size=2, padding=1)
        self.x1_3 = ConvBlock(64 * 3, 64)

        self.up0_1 = nn.Conv2d(64, 32, kernel_size=2, padding=1)
        self.x0_1 = ConvBlock(64, 32)
        self.up0_2 = nn.Conv2d(64, 32, kernel_size=2, padding=1)
        self.x0_2 = ConvBlock(32 * 2, 32)
        self.up0_3 = nn.Conv2d(64, 32, kernel_size=2, padding=1)
        self.x0_3 = ConvBlock(32 * 3, 32)
        self.up0_4 = nn.Conv2d(64, 32, kernel_size=2, padding=1)
        self.x0_4 = ConvBlock(32 * 4, 32)

        # Output
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0_0 = self.x0_0(x)
        x1_0 = self.x1_0(self.pool(x0_0))
        x2_0 = self.x2_0(self.pool(x1_0))
        x3_0 = self.x3_0(self.pool(x2_0))
        x4_0 = self.x4_0(self.pool(x3_0))
        x4_0 = self.dropout(x4_0)

        # Decoder with nested skips
        # Level 3
        up3_1 = F.interpolate(x4_0, scale_factor=2, mode="bilinear", align_corners=True)
        x3_1 = self.x3_1(torch.cat([x3_0, up3_1], dim=1))

        # Level 2
        up2_1 = F.interpolate(x3_0, scale_factor=2, mode="bilinear", align_corners=True)
        x2_1 = self.x2_1(torch.cat([x2_0, up2_1], dim=1))

        up2_2 = F.interpolate(x3_1, scale_factor=2, mode="bilinear", align_corners=True)
        x2_2 = self.x2_2(torch.cat([x2_0, x2_1, up2_2], dim=1))

        # Level 1
        up1_1 = F.interpolate(x2_0, scale_factor=2, mode="bilinear", align_corners=True)
        x1_1 = self.x1_1(torch.cat([x1_0, up1_1], dim=1))

        up1_2 = F.interpolate(x2_1, scale_factor=2, mode="bilinear", align_corners=True)
        x1_2 = self.x1_2(torch.cat([x1_0, x1_1, up1_2], dim=1))

        up1_3 = F.interpolate(x2_2, scale_factor=2, mode="bilinear", align_corners=True)
        x1_3 = self.x1_3(torch.cat([x1_0, x1_1, x1_2, up1_3], dim=1))

        # Level 0
        up0_1 = F.interpolate(x1_0, scale_factor=2, mode="bilinear", align_corners=True)
        x0_1 = self.x0_1(torch.cat([x0_0, up0_1], dim=1))

        up0_2 = F.interpolate(x1_1, scale_factor=2, mode="bilinear", align_corners=True)
        x0_2 = self.x0_2(torch.cat([x0_0, x0_1, up0_2], dim=1))

        up0_3 = F.interpolate(x1_2, scale_factor=2, mode="bilinear", align_corners=True)
        x0_3 = self.x0_3(torch.cat([x0_0, x0_1, x0_2, up0_3], dim=1))

        up0_4 = F.interpolate(x1_3, scale_factor=2, mode="bilinear", align_corners=True)
        x0_4 = self.x0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, up0_4], dim=1))

        out = self.final(x0_4)
        out = F.softmax(out, dim=1)  
        return out

