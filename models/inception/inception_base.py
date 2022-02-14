import torch
import torch.nn as nn


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y=0):
        x = self.conv(x)
        x = self.bn(x) + y
        x = self.relu(x)
        return x


class Stem(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = BasicConv2d(in_planes=3, out_planes=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = BasicConv2d(in_planes=32, out_planes=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = BasicConv2d(in_planes=64, out_planes=128, kernel_size=3, stride=1, padding=1)

        #         self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #         self.branch2 = BasicConv2d(in_planes=64, out_planes=96, kernel_size=3, stride=2, padding=1)

        self.branch3 = nn.Sequential(
            BasicConv2d(in_planes=128, out_planes=64, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_planes=64, out_planes=64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(in_planes=64, out_planes=64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(in_planes=64, out_planes=192, kernel_size=3, stride=1, padding=1)
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(in_planes=128, out_planes=64, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_planes=64, out_planes=192, kernel_size=3, stride=1, padding=1)
        )

    #         self.branch5 = BasicConv2d(in_planes=192, out_planes=192, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        #         x1 = self.branch1(x)
        #         x2 = self.branch2(x)
        #         x = torch.cat([x1, x2], dim=1)

        x1 = self.branch3(x)
        x2 = self.branch4(x)
        x = torch.cat([x1, x2], dim=1)

        #         x1 = self.branch1(x)
        #         x2 = self.branch5(x)
        #         x = torch.cat([x1, x2], dim=1)

        return x
