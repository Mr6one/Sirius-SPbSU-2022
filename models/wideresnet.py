import torch.nn as nn
from .base_model import BaseModel


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.apply_downsample = in_planes != out_planes

        if self.apply_downsample:
            self.conv_shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        skip = x

        x = self.bn1(x)
        x = self.relu(x)

        if self.apply_downsample:
            skip = self.conv_shortcut(x)

        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)

        return skip + x


class WideResNet(BaseModel):
    def __init__(self, depth, num_classes, widen_factor=1):
        super().__init__()

        assert depth > 4
        assert (depth - 4) % 6 == 0
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        n = (depth - 4) // 6

        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.block1 = self.make_layer(n, n_channels[0], n_channels[1], 1)
        self.block2 = self.make_layer(n, n_channels[1], n_channels[2], 2)
        self.block3 = self.make_layer(n, n_channels[2], n_channels[3], 2)

        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)

        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(n_channels[3], num_classes)

    def make_layer(self, n_blocks, in_channels, out_channels, stride):
        layer = [BasicBlock(in_channels, out_channels, stride)]

        for _ in range(1, n_blocks):
            layer.append(BasicBlock(out_channels, out_channels, 1))

        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(self.bn1(x))

        x = self.avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x


def wide_resnet_28x2(num_classes=10):
    return WideResNet(depth=28, widen_factor=2, num_classes=num_classes)


def wide_resnet_28x10(num_classes=10):
    return WideResNet(depth=28, widen_factor=10, num_classes=num_classes)


def wide_resnet_28x12(num_classes=10):
    return WideResNet(depth=28, widen_factor=12, num_classes=num_classes)


def wide_resnet_16x8(num_classes=10):
    return WideResNet(depth=16, widen_factor=8, num_classes=num_classes)


def wide_resnet_40x8(num_classes=10):
    return WideResNet(depth=40, widen_factor=8, num_classes=num_classes)
