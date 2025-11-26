import torch.nn as nn

class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, conv_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(conv_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(conv_channels, conv_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(conv_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(2, 2)

    def forward(self, input_batch):
        # Block 1
        block_out = self.conv1(input_batch)
        block_out = self.bn1(block_out)
        block_out = self.relu1(block_out)

        # Block 2
        block_out = self.conv2(block_out)
        block_out = self.bn2(block_out)
        block_out = self.relu2(block_out)

        return self.maxpool(block_out)


class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels * 2)
        self.block3 = LunaBlock(conv_channels * 2, conv_channels * 4)
        self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8)

        final_flatten_size = (conv_channels * 8) * 3 * 3 * 3
        self.linear = nn.Linear(final_flatten_size, 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m , nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # (B, 1, 48, 48, 48)
        out = self.block1(x) # (B, 8, 24, 24, 24)
        out = self.block2(out) # (B, 16, 12, 12, 12)
        out = self.block3(out) # (B, 32, 6, 6, 6)
        out = self.block4(out) # (B, 64, 3, 3, 3)

        # Flatten
        out = out.view(out.size(0), -1) # (B, 1728)

        # FC
        logits = self.linear(out)

        return logits
