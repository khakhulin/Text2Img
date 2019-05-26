import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_planes, out_planes):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=False)


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=True)


class Interpolate(nn.Module):
    """
    Interpolate initial up into scale_factor
    """

    def __init__(self, scale_factor, mode, size=None):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.size = size

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, size=self.size)
        return x


class UpBlock(nn.Sequential):
    def __init__(self, in_planes, out_planes):
        super().__init__(nn.Sequential(
            Interpolate(scale_factor=2, mode='nearest'),
            conv3x3(in_planes, out_planes * 2),
            nn.BatchNorm2d(out_planes * 2),
            nn.GLU(dim=1)
        ))


# Keep the spatial size

# class GluConv(nn.Sequential):
#     def __init__(self, in_planes, out_planes):
#         super().__init__(nn.Sequential(
#             conv3x3(in_planes, out_planes * 2),
#             nn.BatchNorm2d(out_planes * 2),
#             nn.GLU()
#         ))


class LeakyConv3x3(nn.Sequential):
    def __init__(self, in_planes, out_planes):
        super().__init__(nn.Sequential(
            conv3x3(in_planes, out_planes),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace=True)
        ))


class LeakyConv(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super().__init__(nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace=True)
        ))
