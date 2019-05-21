import torch
import torch.nn as nn

from nn_utils import *
from utils import *


class Discriminator(nn.Module):
    """
    Base module which compute score for condition
    and uncoditioned setting.
    Return probability of the real/fake image
    """
    def __init__(self, ndf, encoder_dim, condition=False):
        super(Discriminator, self).__init__()
        self.df_dim = ndf
        self.encoder_dim = encoder_dim
        self.condition = condition
        if self.condition:
            # to eliminate condition part
            self.conv_embedder = LeakyConv3x3(ndf * 8 + encoder_dim, ndf * 8)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        if self.condition and c_code is not None:
            # conditioning output
            c_code = c_code.view(-1, self.encoder_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # (ngf+egf) x 4 x 4
            h_c_code = torch.cat((h_code, c_code), 1)
            # ngf x in_size x in_size
            h_c_code = self.conv_embedder.forward(h_c_code)
        else:
            h_c_code = h_code

        output = self.logits(h_c_code)
        return output.view(-1)


class Discriminator64(nn.Module):
    def __init__(self, dim, embd_dim, condition=False):
        super(Discriminator64, self).__init__()
        self.ndf = dim
        self.embd_dim = embd_dim
        self.condition = condition
        self.img_code_s16_func = Downsample16(dim)
        if not condition:
            self.uncond_discriminator = Discriminator(dim, embd_dim, condition=False)

        self.cond_discriminator = Discriminator(dim, embd_dim, condition=True)

    def forward(self, x_var):
        # 4 x 4 x 8 df
        x_code4 = self.img_code_s16_func.forward(x_var)
        return x_code4


class Discriminator128(Discriminator64):
    def __init__(self, ndf, embd_dim, condition=False):
        super(Discriminator128, self).__init__(ndf, embd_dim, condition)
        self.img_code_s32_1 = LeakyConv(ndf * 8, ndf * 16, 4, 2, 1)
        self.img_code_s32_2 = LeakyConv3x3(ndf * 16, ndf * 8)

    def forward(self, x_var):
        # 8 x 8 x 8df -> 4 x 4 x 16df -> 4 x 4 x 8df
        x_code8 = self.img_code_s16_func.forward(x_var)
        x_code4 = self.img_code_s32_1.forward(x_code8)
        x_code4 = self.img_code_s32_2.forward(x_code4)
        return x_code4


class Discriminator256(Discriminator128):
    def __init__(self, ndf, embd_dim, condition=False):
        super(Discriminator256, self).__init__(ndf, embd_dim, condition)
        self.image_code = nn.Sequential(
            self.img_code_s16_func,
            self.img_code_s32_1,
            LeakyConv(ndf * 16, ndf * 32, 4, 2, 1),
            LeakyConv3x3(ndf * 32, ndf * 16),
            LeakyConv3x3(ndf * 16, ndf * 8),
        )

    def forward(self, x_var):
        x_code4 = self.image_code(x_var)
        return x_code4

if __name__ == '__main__':
    print(Discriminator256(10, 64).cond_discriminator)