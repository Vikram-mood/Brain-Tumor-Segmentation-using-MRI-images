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


# ------------------------------
# Attention Gate
# ------------------------------
class AttentionGate(nn.Module):
    def __init__(self, in_ch, gating_ch, inter_ch):
        super(AttentionGate, self).__init__()
        # 1x1 conv for skip connection features
        self.theta_x = nn.Conv2d(in_ch, inter_ch, kernel_size=1, stride=1, padding=0)
        # 1x1 conv for gating signal
        self.phi_g = nn.Conv2d(gating_ch, inter_ch, kernel_size=1, stride=1, padding=0)
        # Final conv for attention coefficients
        self.psi = nn.Conv2d(inter_ch, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x, g):
        # Transform skip features
        theta_x = self.theta_x(x)
        # Transform gating features
        phi_g = self.phi_g(g)

        # Resize gating signal if needed
        if theta_x.shape[2:] != phi_g.shape[2:]:
            phi_g = F.interpolate(phi_g, size=theta_x.shape[2:], mode="bilinear", align_corners=True)

        # Add + ReLU
        add_xg = F.relu(theta_x + phi_g, inplace=True)

        # 1x1 conv + sigmoid for attention mask
        psi = torch.sigmoid(self.psi(add_xg))

        # Apply attention coefficients
        out = x * psi
        return out


# ------------------------------
# Attention U-Net
# ------------------------------
class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, dropout=0.5):
        super(AttentionUNet, self).__init__()

        # Encoder
        self.conv1 = ConvBlock(in_channels, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)
        self.conv5 = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)

        # Attention gates
        self.attn7 = AttentionGate(in_ch=256, gating_ch=256, inter_ch=128)
        self.attn8 = AttentionGate(in_ch=128, gating_ch=128, inter_ch=64)
        self.attn9 = AttentionGate(in_ch=64, gating_ch=64, inter_ch=32)
        self.attn10 = AttentionGate(in_ch=32, gating_ch=32, inter_ch=16)

        # Decoder
        self.up7 = nn.Conv2d(512, 256, kernel_size=2, padding=0)
        self.conv7 = ConvBlock(512, 256)

        self.up8 = nn.Conv2d(256, 128, kernel_size=2, padding=0)
        self.conv8 = ConvBlock(256, 128)

        self.up9 = nn.Conv2d(128, 64, kernel_size=2, padding=0)
        self.conv9 = ConvBlock(128, 64)

        self.up10 = nn.Conv2d(64, 32, kernel_size=2, padding=0)
        self.conv10 = ConvBlock(64, 32)

        # Final output
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool(c1)

        c2 = self.conv2(p1)
        p2 = self.pool(c2)

        c3 = self.conv3(p2)
        p3 = self.pool(c3)

        c4 = self.conv4(p3)
        p4 = self.pool(c4)

        c5 = self.conv5(p4)
        c5 = self.dropout(c5)

        # Decoder with attention
        up7 = F.interpolate(c5, scale_factor=2, mode="bilinear", align_corners=True)
        attn7 = self.attn7(c4, up7)
        c7 = self.conv7(torch.cat([attn7, up7], dim=1))

        up8 = F.interpolate(c7, scale_factor=2, mode="bilinear", align_corners=True)
        attn8 = self.attn8(c3, up8)
        c8 = self.conv8(torch.cat([attn8, up8], dim=1))

        up9 = F.interpolate(c8, scale_factor=2, mode="bilinear", align_corners=True)
        attn9 = self.attn9(c2, up9)
        c9 = self.conv9(torch.cat([attn9, up9], dim=1))

        up10 = F.interpolate(c9, scale_factor=2, mode="bilinear", align_corners=True)
        attn10 = self.attn10(c1, up10)
        c10 = self.conv10(torch.cat([attn10, up10], dim=1))

        out = self.final(c10)
        out = F.softmax(out, dim=1)  
        return out

