# sub-parts of the U-Net model

# import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv3x3(in_planes, out_planes, stride = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1) # , bias=False


class Basic_Block(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groupnorm=False):
        super(Basic_Block, self).__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.GroupNorm(1, planes)
        self.bn1 = nn.BatchNorm2d(planes) if not groupnorm else nn.GroupNorm(8, planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        # self.bn2 = nn.GroupNorm(1, planes)
        self.bn2 = nn.BatchNorm2d(planes) if not groupnorm else nn.GroupNorm(8, planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, groupnorm=False):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            # nn.GroupNorm(1, out_ch),
            nn.BatchNorm2d(out_ch) if not groupnorm else nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            # nn.GroupNorm(1, out_ch),
            nn.BatchNorm2d(out_ch) if not groupnorm else nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, bridge_ch, out_ch, groupnorm=False):
        super(up, self).__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3, stride = 2, padding = 1, output_padding = 1),
            # nn.GroupNorm(1, out_ch),
            nn.BatchNorm2d(out_ch) if not groupnorm else nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True)
        )

        self.bridge = nn.Sequential(
            nn.Conv2d(bridge_ch, out_ch, 1),
            # nn.GroupNorm(1, out_ch),
            nn.BatchNorm2d(out_ch) if not groupnorm else nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.bridge(x2)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (0, diffY, 0, diffX)) # Harry
        x1 += x2
        return x1


class single_up(nn.Module):
    def __init__(self, in_ch, out_ch, groupnorm=False):
        super(single_up, self).__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3, stride = 2, padding = 1, output_padding = 1),
            # nn.GroupNorm(1, out_ch),
            nn.BatchNorm2d(out_ch) if not groupnorm else nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (0, diffY, 0, diffX))  # Harry
        return x1


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Sequential(
                double_conv(in_ch, in_ch),
                nn.Conv2d(in_ch, out_ch, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x