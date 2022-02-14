import torch
import torch.nn as nn
from .inception_base import *
from ..base_model import BaseModel


class InceptionA(nn.Module):
    def __init__(self):
        super().__init__()

        self.branch1 = BasicConv2d(in_planes=384, out_planes=96, kernel_size=1, stride=1, padding=0)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_planes=384, out_planes=64, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_planes=64, out_planes=96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_planes=384, out_planes=64, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_planes=64, out_planes=96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_planes=96, out_planes=96, kernel_size=3, stride=1, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(in_planes=384, out_planes=96, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        return x


class InceptionB(nn.Module):
    def __init__(self):
        super().__init__()

        self.branch1 = BasicConv2d(in_planes=1024, out_planes=384, kernel_size=1, stride=1, padding=0)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_planes=1024, out_planes=192, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_planes=192, out_planes=224, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(in_planes=224, out_planes=256, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_planes=1024, out_planes=192, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_planes=192, out_planes=192, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(in_planes=192, out_planes=224, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(in_planes=224, out_planes=224, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(in_planes=224, out_planes=256, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(in_planes=1024, out_planes=128, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        return x


class InceptionC(nn.Module):
    def __init__(self):
        super().__init__()

        self.branch1 = BasicConv2d(in_planes=1536, out_planes=256, kernel_size=1, stride=1, padding=0)

        self.branch2 = BasicConv2d(in_planes=1536, out_planes=384, kernel_size=1, stride=1, padding=0)
        self.branch2_extention1 = BasicConv2d(in_planes=384, out_planes=256, kernel_size=(1, 3), stride=1,
                                              padding=(0, 1))
        self.branch2_extention2 = BasicConv2d(in_planes=384, out_planes=256, kernel_size=(3, 1), stride=1,
                                              padding=(1, 0))

        self.branch3 = nn.Sequential(
            BasicConv2d(in_planes=1536, out_planes=384, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_planes=384, out_planes=448, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(in_planes=448, out_planes=512, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )
        self.branch3_extention1 = BasicConv2d(in_planes=512, out_planes=256, kernel_size=(3, 1), stride=1,
                                              padding=(1, 0))
        self.branch3_extention2 = BasicConv2d(in_planes=512, out_planes=256, kernel_size=(1, 3), stride=1,
                                              padding=(0, 1))

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(in_planes=1536, out_planes=256, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x1 = self.branch1(x)

        x2 = self.branch2(x)
        x2_1 = self.branch2_extention1(x2)
        x2_2 = self.branch2_extention2(x2)

        x3 = self.branch3(x)
        x3_1 = self.branch3_extention1(x3)
        x3_2 = self.branch3_extention2(x3)

        x4 = self.branch4(x)
        x = torch.cat([x1, x2_1, x2_2, x3_1, x3_2, x4], dim=1)

        return x


class ReductionA(nn.Module):
    def __init__(self):
        super().__init__()

        self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.branch2 = BasicConv2d(in_planes=384, out_planes=384, kernel_size=3, stride=2, padding=1)

        self.branch3 = nn.Sequential(
            BasicConv2d(in_planes=384, out_planes=192, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_planes=192, out_planes=224, kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_planes=224, out_planes=256, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x = torch.cat([x1, x2, x3], dim=1)

        return x


class ReductionB(nn.Module):
    def __init__(self):
        super().__init__()

        self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_planes=1024, out_planes=192, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_planes=192, out_planes=192, kernel_size=3, stride=2, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_planes=1024, out_planes=256, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_planes=256, out_planes=256, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(in_planes=256, out_planes=320, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(in_planes=320, out_planes=320, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x = torch.cat([x1, x2, x3], dim=1)

        return x


class InceptionV4(BaseModel):
    def __init__(self, n_blocks_A=4, n_blocks_B=7, n_blocks_C=3, num_classes=10):
        super().__init__()

        self.stem = Stem()

        self.inception_A = nn.Sequential(*[
            InceptionA() for _ in range(n_blocks_A)
        ])

        self.reduction_A = ReductionA()

        self.inception_B = nn.Sequential(*[
            InceptionB() for _ in range(n_blocks_B)
        ])

        self.reduction_B = ReductionB()

        self.inception_C = nn.Sequential(*[
            InceptionC() for _ in range(n_blocks_C)
        ])

        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.stem(x)

        x = self.inception_A(x)
        x = self.reduction_A(x)
        x = self.inception_B(x)
        x = self.reduction_B(x)
        x = self.inception_C(x)

        x = self.avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x


def inception_v4(num_classes=10):
    return InceptionV4(num_classes=num_classes)
