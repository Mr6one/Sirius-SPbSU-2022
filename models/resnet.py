import torch.nn as nn
from .base_model import BaseModel


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, apply_downsample=False):
        super().__init__()

        self.apply_downsample = apply_downsample

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if apply_downsample:
            self.downsample = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            ])

    def forward(self, x):
        skip = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.apply_downsample:
            skip = self.downsample(skip)

        x += skip
        x = self.relu(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, middle_channels, stride=1, apply_downsample=False):
        super().__init__()

        out_channels = middle_channels * self.expansion
        self.apply_downsample = apply_downsample

        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, middle_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.conv3 = nn.Conv2d(middle_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if apply_downsample:
            self.downsample = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            ])

    def forward(self, x):
        skip = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.apply_downsample:
            skip = self.downsample(skip)

        x += skip
        x = self.relu(x)

        return x


class ResNetBase(BaseModel):
    def __init__(self, block, blocks, n_classes):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.layer1 = self.make_layer(block, blocks[0], 64, stride=1)
        self.layer2 = self.make_layer(block, blocks[1], 128, stride=2)
        self.layer3 = self.make_layer(block, blocks[2], 256, stride=2)
        self.layer4 = self.make_layer(block, blocks[3], 512, stride=2)

        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512 * block.expansion, n_classes)

    def make_layer(self, block, n_blocks, middle_channels, stride):
        layer = [block(self.inplanes, middle_channels, stride, apply_downsample=True)]
        self.inplanes = middle_channels * block.expansion

        for _ in range(1, n_blocks):
            layer.append(block(self.inplanes, middle_channels))

        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x


def resnet18(n_classes=10):
    return ResNetBase(BasicBlock, [2, 2, 2, 2], n_classes)


def resnet34(n_classes=10):
    return ResNetBase(BasicBlock, [3, 4, 6, 3], n_classes)


def resnet50(n_classes=10):
    return ResNetBase(Bottleneck, [3, 4, 6, 3], n_classes)


def resnet101(n_classes=10):
    return ResNetBase(Bottleneck, [3, 4, 23, 3], n_classes)


def resnet152(n_classes=10):
    return ResNetBase(Bottleneck, [3, 8, 36, 3], n_classes)
