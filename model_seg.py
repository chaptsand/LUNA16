import torch
import torch.nn as nn

class LunaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1,bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        return out

class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=1, start_channels=16):
        super().__init__()

        self.n_classes = n_classes

        # --- Encoder ---
        self.enc1 = LunaBlock(in_channels, start_channels) # 1 -> 16
        self.pool1 = nn.MaxPool3d(2, 2) # (16, 48, 48, 48) -> (16, 24, 24, 24)

        self.enc2 = LunaBlock(start_channels, start_channels * 2) # 16 -> 32
        self.pool2 = nn.MaxPool3d(2, 2) # (32, 24, 24, 24) -> (32, 12, 12 ,12)

        self.enc3 = LunaBlock(start_channels * 2, start_channels * 4) # 32 -> 64
        self.pool3 = nn.MaxPool3d(2, 2) # (64, 12, 12 ,12) -> (64, 6, 6 ,6)

        self.bottleneck = LunaBlock(start_channels * 4, start_channels * 8) # 64 -> 128

        # --- Decoder ---
        # Up3
        self.up3 = nn.ConvTranspose3d(start_channels * 8, start_channels * 4, kernel_size=2, stride=2)
        self.dec3 = LunaBlock(start_channels * 8, start_channels * 4)

        # Up2
        self.up2 = nn.ConvTranspose3d(start_channels * 4, start_channels * 2, kernel_size=2, stride=2)
        self.dec2 = LunaBlock(start_channels * 4, start_channels * 2)

        # Up1
        self.up1 = nn.ConvTranspose3d(start_channels * 2, start_channels, kernel_size=2, stride=2)
        self.dec1 = LunaBlock(start_channels * 2, start_channels)

        # --- Final Output ---
        self.final_conv = nn.Conv3d(start_channels, n_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)

        enc2 = self.pool1(enc1)
        enc2 = self.enc2(enc2)

        enc3 = self.pool2(enc2)
        enc3 = self.enc3(enc3)

        # --- Bottleneck ---
        bottleneck = self.pool3(enc3)
        bottleneck = self.bottleneck(bottleneck)

        # --- Decoder ---
        # Upsample
        up3 = self.up3(bottleneck)
        # Concatenate (Skip Connection)
        cat3 = torch.cat([up3, enc3], dim=1)
        # Convolution fusion
        dec3 = self.dec3(cat3)

        up2 = self.up2(dec3)
        cat2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec2(cat2)

        up1 = self.up1(dec2)
        cat1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec1(cat1)

        return self.final_conv(dec1)
