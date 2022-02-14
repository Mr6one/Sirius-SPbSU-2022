import torch
import torch.nn as nn
from .inception_base import *
from ..base_model import BaseModel


class InceptionResnetA(nn.Module):
    def __init__(self):
        super().__init__()

        self.branch1 = BasicConv2d(in_planes=384, out_planes=32, kernel_size=1, stride=1, padding=0)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_planes=384, out_planes=32, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_planes=32, out_planes=32, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_planes=384, out_planes=32, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_planes=32, out_planes=48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_planes=48, out_planes=64, kernel_size=3, stride=1, padding=1),
        )

        self.conv_with_skip = BasicConv2d(in_planes=128, out_planes=384, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = torch.cat([x1, x2, x3], dim=1)

        x = self.conv_with_skip(x4, x)

        return x


class InceptionResnetB(nn.Module):
    def __init__(self):
        super().__init__()

        self.branch1 = BasicConv2d(in_planes=1152, out_planes=192, kernel_size=1, stride=1, padding=0)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_planes=1152, out_planes=128, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_planes=128, out_planes=160, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(in_planes=160, out_planes=192, kernel_size=(7, 1), stride=1, padding=(3, 0)),
        )

        self.conv_with_skip = BasicConv2d(in_planes=384, out_planes=1152, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = torch.cat([x1, x2], dim=1)

        x = self.conv_with_skip(x3, x)

        return x


class InceptionResnetC(nn.Module):
    def __init__(self):
        super().__init__()

        self.branch1 = BasicConv2d(in_planes=2144, out_planes=192, kernel_size=1, stride=1, padding=0)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_planes=2144, out_planes=192, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_planes=192, out_planes=224, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(in_planes=224, out_planes=256, kernel_size=(3, 1), stride=1, padding=(1, 0)),
        )

        self.conv_with_skip = BasicConv2d(in_planes=448, out_planes=2144, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = torch.cat([x1, x2], dim=1)

        x = self.conv_with_skip(x3, x)

        return x


class ReductionResnetA(nn.Module):
    def __init__(self):
        super().__init__()

        self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.branch2 = BasicConv2d(in_planes=384, out_planes=384, kernel_size=3, stride=2, padding=1)

        self.branch3 = nn.Sequential(
            BasicConv2d(in_planes=384, out_planes=256, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_planes=256, out_planes=256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_planes=256, out_planes=384, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x = torch.cat([x1, x2, x3], dim=1)

        return x


class ReductionResnetB(nn.Module):
    def __init__(self):
        super().__init__()

        self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_planes=1152, out_planes=256, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_planes=256, out_planes=384, kernel_size=3, stride=2, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_planes=1152, out_planes=256, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_planes=256, out_planes=288, kernel_size=3, stride=2, padding=1)
        )

        self.branch4 = nn.Sequential(
            BasicConv2d(in_planes=1152, out_planes=256, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_planes=256, out_planes=288, kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_planes=288, out_planes=320, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        return x


class InceptionResnetV2(BaseModel):
    def __init__(self, n_blocks_A=5, n_blocks_B=10, n_blocks_C=5, num_classes=10):
        super().__init__()

        self.stem = Stem()

        self.inception_resnet_A = nn.Sequential(*[
            InceptionResnetA() for _ in range(n_blocks_A)
        ])

        self.reduction_A = ReductionResnetA()

        self.inception_resnet_B = nn.Sequential(*[
            InceptionResnetB() for _ in range(n_blocks_B)
        ])

        self.reduction_B = ReductionResnetB()

        self.inception_resnet_C = nn.Sequential(*[
            InceptionResnetC() for _ in range(n_blocks_C)
        ])

        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(2144, num_classes)

    def forward(self, x):
        x = self.stem(x)

        x = self.inception_resnet_A(x)
        x = self.reduction_A(x)
        x = self.inception_resnet_B(x)
        x = self.reduction_B(x)
        x = self.inception_resnet_C(x)

        x = self.avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x


def inception_resnet_v2(num_classes=10):
    return InceptionResnetV2(num_classes=num_classes)
