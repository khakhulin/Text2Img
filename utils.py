import torch.nn as nn

from nn_utils import LeakyConv


class Downsample16(nn.Sequential):
    def __init__(self, num_out):
        super().__init__(nn.Sequential(
        # --> state size. ndf x in_size/2 x in_size/2
        nn.Conv2d(3, num_out, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 2ndf x x in_size/4 x in_size/4
        LeakyConv(num_out, num_out * 2, 4, 2, 1),
        # --> state size 4ndf x in_size/8 x in_size/8
        LeakyConv(num_out * 2, num_out * 4, 4, 2, 1),
        # --> state size 8ndf x in_size/16 x in_size/16
        LeakyConv(num_out * 4, num_out * 8, 4, 2, 1)
        )
    )

if __name__ == '__main__':
    print(Downsample16(10))