import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34
from utils import resize_input

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionResNetUNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=3, pretrained=True):
        super(AttentionResNetUNet, self).__init__()
        self.encoder = resnet34(pretrained=pretrained)
        self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Encoder layers
        self.enc1 = self.encoder.conv1
        self.bn1 = self.encoder.bn1
        self.relu = self.encoder.relu
        self.maxpool = self.encoder.maxpool
        self.enc2 = self.encoder.layer1
        self.enc3 = self.encoder.layer2
        self.enc4 = self.encoder.layer3
        self.enc5 = self.encoder.layer4

        # Decoder
        self.center = self._make_conv_block(512, 512)
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.decoder4 = self._make_conv_block(256 + 256, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.decoder3 = self._make_conv_block(128 + 128, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.decoder2 = self._make_conv_block(64 + 64, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = self._make_conv_block(32, 32)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if x.shape[2:] != (224, 224):
            x = resize_input(x, target_size=(224, 224))

        # Encoder
        x1 = self.relu(self.bn1(self.enc1(x)))
        x1_pool = self.maxpool(x1)
        x2 = self.enc2(x1_pool)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        # Decoder with attention
        x = self.center(x5)
        x = self.upconv4(x)
        x4_att = self.att4(g=x, x=x4)
        x = torch.cat([x, x4_att], dim=1)
        x = self.decoder4(x)
        x = self.upconv3(x)
        x3_att = self.att3(g=x, x=x3)
        x = torch.cat([x, x3_att], dim=1)
        x = self.decoder3(x)
        x = self.upconv2(x)
        x2_att = self.att2(g=x, x=x2)
        x = torch.cat([x, x2_att], dim=1)
        x = self.decoder2(x)
        x = self.upconv1(x)
        x = self.decoder1(x)
        x = self.final_conv(x)
        return x