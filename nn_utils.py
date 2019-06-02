import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SpectralNorm


def conv1x1(in_planes, out_planes):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=False)


def conv3x3(in_planes, out_planes, is_spectral=False):
    "3x3 convolution with padding"
    if is_spectral:
        return SpectralNorm(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                         padding=1, bias=True))
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


class Downsample16(nn.Sequential):
    def __init__(self, num_out):
        super().__init__(nn.Sequential(
        nn.Conv2d(3, num_out, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        LeakyConv(num_out, num_out * 2, 4, 2, 1),
        LeakyConv(num_out * 2, num_out * 4, 4, 2, 1),
        LeakyConv(num_out * 4, num_out * 8, 4, 2, 1)
        )
    )


class SDownsample16(nn.Sequential):
    def __init__(self, num_out):
        super().__init__(nn.Sequential(
        SpectralNorm(nn.Conv2d(3, num_out, 4, 2, 1, bias=False)),
        nn.LeakyReLU(0.2, inplace=True),
        SpectralLeakyConv(num_out, num_out * 2, 4, 2, 1),
        SpectralLeakyConv(num_out * 2, num_out * 4, 4, 2, 1),
        SpectralLeakyConv(num_out * 4, num_out * 8, 4, 2, 1)
        )
    )


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
    def __init__(self, in_planes, out_planes, is_spectral=False):
        super().__init__(nn.Sequential(
            Interpolate(scale_factor=2, mode='nearest'),
            conv3x3(in_planes, out_planes * 2, is_spectral),
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
    def __init__(self, in_planes, out_planes, is_spectral=False):
        super().__init__(nn.Sequential(
            conv3x3(in_planes, out_planes, is_spectral),
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


class SpectralLeakyConv(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super().__init__(nn.Sequential(
            SpectralNorm(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=False)),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace=True)
        ))