import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, dropout=0.5):
        super(UNet, self).__init__()

        # Encoder
        self.conv1 = self.double_conv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = self.double_conv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = self.double_conv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = self.double_conv(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.conv5 = self.double_conv(256, 512)
        self.drop5 = nn.Dropout(dropout)

        # Decoder
        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv6 = self.double_conv(512, 256)

        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7 = self.double_conv(256, 128)

        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8 = self.double_conv(128, 64)

        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv9 = self.double_conv(64, 32)

        # Final output
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def double_conv(self, in_ch, out_ch):
       
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        # Bottleneck
        c5 = self.conv5(p4)
        c5 = self.drop5(c5)

        # Decoder
        u6 = self.up6(c5)
        u6 = torch.cat([c4, u6], dim=1)
        c6 = self.conv6(u6)

        u7 = self.up7(c6)
        u7 = torch.cat([c3, u7], dim=1)
        c7 = self.conv7(u7)

        u8 = self.up8(c7)
        u8 = torch.cat([c2, u8], dim=1)
        c8 = self.conv8(u8)

        u9 = self.up9(c8)
        u9 = torch.cat([c1, u9], dim=1)
        c9 = self.conv9(u9)

        out = self.out_conv(c9)
        out = F.softmax(out, dim=1) 

        return out

