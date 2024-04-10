import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_builder.pmnet_v3 import _ResLayer, _ASPP, _ConvBnReLU, ConRu, ConRuT


class _Stem(nn.Sequential):
    """
    The 1st conv layer.
    Note that the max pooling is different from both MSRA and FAIR ResNet.
    """

    def __init__(self, out_ch, in_ch=2):
        super(_Stem, self).__init__()
        self.add_module("conv1", _ConvBnReLU(in_ch, out_ch, 7, 2, 3, 1))
        self.add_module("pool", nn.MaxPool2d(2, 2, 1, ceil_mode=True))


class RewardModel(nn.Module):

    def __init__(self, n_blocks, atrous_rates, multi_grids, output_stride, output_dim=32):
        super().__init__()

        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]

        # Encoder
        ch = [64 * 2 ** p for p in range(6)]
        self.layer1 = _Stem(ch[0], in_ch=1)
        self.layer2 = _ResLayer(n_blocks[0], ch[0], ch[2], s[0], d[0])
        self.layer3 = _ResLayer(n_blocks[1], ch[2], ch[3], s[1], d[1])
        self.layer4 = _ResLayer(n_blocks[2], ch[3], ch[3], s[2], d[2])
        self.layer5 = _ResLayer(n_blocks[3], ch[3], ch[4], s[3], d[3], multi_grids)
        self.aspp = _ASPP(ch[4], 256, atrous_rates)
        concat_ch = 256 * (len(atrous_rates) + 2)
        self.add_module("fc1", _ConvBnReLU(concat_ch, 512, 1, 1, 0, 1))
        self.reduce = _ConvBnReLU(256, 256, 1, 1, 0, 1)

        # Decoder
        self.conv_up5 = ConRu(512, 512, 3, 1)
        self.conv_up4 = ConRu(512 + 512, 512, 3, 1)
        self.conv_up3 = ConRuT(512 + 512, 256, 3, 1)
        self.conv_up2 = ConRu(256 + 256, 256, 3, 1)
        self.conv_up1 = ConRu(256 + 256, 256, 3, 1)

        self.conv_up0 = ConRu(256 + 64, 128, 3, 1)
        self.conv_up00 = nn.Sequential(
            nn.Conv2d(128 + 1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1))

        # Adjust output shape
        self.fc2 = nn.Linear(256 ** 2, output_dim ** 2)

    def forward(self, x):
        # Encoder
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.reduce(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.layer5(x5)
        x7 = self.aspp(x6)
        x8 = self.fc1(x7)

        # Decoder
        xup5 = self.conv_up5(x8)
        xup5 = torch.cat([xup5, x5], dim=1)
        xup4 = self.conv_up4(xup5)
        xup4 = torch.cat([xup4, x4], dim=1)
        xup3 = self.conv_up3(xup4)
        xup3 = torch.cat([xup3, x3], dim=1)
        xup2 = self.conv_up2(xup3)
        xup2 = torch.cat([xup2, x2], dim=1)
        xup1 = self.conv_up1(xup2)
        xup1 = torch.cat([xup1, x1], dim=1)
        xup0 = self.conv_up0(xup1)

        xup0 = F.interpolate(xup0, size=x.shape[2:], mode="bilinear", align_corners=False)
        xup0 = torch.cat([xup0, x], dim=1)
        xup00 = self.conv_up00(xup0)

        # output layer
        out_flatten = xup00.view(-1, 256 ** 2)
        out = F.relu(self.fc2(out_flatten))

        return out


class SimpleCNN(nn.Module):
    def __init__(self, output_dim: int = 1024):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 32 * 32, 1024)
        self.fc2 = nn.Linear(1024, output_dim)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Convolutional layers with ReLU activation and max pooling
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))

        # Flatten the tensor for the fully connected layers
        x = torch.flatten(x, start_dim=1)

        # Fully connected layers with ReLU activation and dropout
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = nn.ReLU()(self.fc2(x))
        x = self.dropout(x)

        return x
