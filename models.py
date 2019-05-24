from nn_utils import *
from utils import *
from global_attention import GlobalAttentionGeneral


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            nn.GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


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


class ConditionNoise(nn.Module):
    def __init__(self, embd_dim, condition_dim, device):
        super(ConditionNoise, self).__init__()
        self.t_dim = embd_dim
        self.c_dim = condition_dim
        self.device = device
        self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        self.relu = nn.ReLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size(), requires_grad=True).normal_().to(self.device)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class UpGenMode(nn.Module):
    def __init__(self, ngf, z_dim, text_embd_dim):
        super(UpGenMode, self).__init__()
        self.gf_dim = ngf
        self.in_dim = z_dim + text_embd_dim

        nz, ngf = self.in_dim, self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            nn.GLU())

        self.upsample = nn.Sequential(
            UpBlock(ngf, ngf // 2),
            UpBlock(ngf // 2, ngf // 4),
            UpBlock(ngf // 4, ngf // 8),
            UpBlock(ngf // 8, ngf // 16)
        )

    def forward(self, z_code, c_code):
        """
        :param z_code: batch x x
        :param c_code: batch x text_embd_dim
        :return: batch x ngf/16 x 64 x 64
        """
        #  ngf x 4 x 4 ->
        #  ngf/3 x 8 x 8 -> ngf/16 x 64 x 64
        c_z_code = torch.cat((c_code, z_code), 1)
        # state size
        out_code = self.fc(c_z_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # state size ngf/3 x 8 x 8
        out_code = self.upsample(out_code)
        return out_code


class ImageGenMod(nn.Module):
    def __init__(self, ngf):
        super(ImageGenMod, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class AttentionGenerator(nn.Module):
    def __init__(self, ngf, nef, text_embd_dim, num_res_block):
        super(AttentionGenerator, self).__init__()
        self.gf_dim = ngf
        self.encoder_dim = nef
        self.cf_dim = text_embd_dim
        self.num_residual = num_res_block
        layers = nn.ModuleList()
        self.res_layers = [layers.append(ResBlock(ngf * 2)) for _ in range(self.num_residual)]
        self.att = GlobalAttentionGeneral(ngf, self.encoder_dim)
        self.upsample = UpBlock(ngf * 2, ngf)

    def forward(self, h_code, c_code, word_embs, mask):
        self.att.applyMask(mask)
        c_code, att = self.att(h_code, word_embs)
        h_c_code = torch.cat((h_code, c_code), 1)
        out_code = self.residual(h_c_code)
        # state size ngf/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)

        return out_code, att


class Generator(nn.Module):
    def __init__(self, ngf, nef, ncf, branch_num, device, z_dim, num_res_block=10):
        super(Generator, self).__init__()
        self.ca_net = ConditionNoise(nef, ncf, device)
        self.branch_num = branch_num
        if branch_num > 0:
            self.h_net1 = UpGenMode(ngf * 16, z_dim=z_dim, text_embd_dim=ncf)
            self.img_net1 = ImageGenMod(ngf)
        if branch_num > 1:
            self.h_net2 = AttentionGenerator(ngf, nef, ncf, num_res_block=num_res_block)
            self.img_net2 = ImageGenMod(ngf)
        if branch_num > 2:
            self.h_net3 = AttentionGenerator(ngf, nef, ncf, num_res_block=num_res_block)
            self.img_net3 = ImageGenMod(ngf)

    def forward(self, z_code, sent_emb, word_embs, mask):
        """
        :param z_code: batch x z_dim
        :param sent_emb: batch x text_embd_dim
        :param word_embs: batch x cdf x seq_len
        :param mask: batch x seq_len
        :return: fake_imgs, att_maps, mu, logvar
        """
        fake_imgs = []
        att_maps = []
        c_code, mu, logvar = self.ca_net(sent_emb)

        if self.branch_num > 0:
            h_code1 = self.h_net1(z_code, c_code)
            fake_img1 = self.img_net1(h_code1)
            fake_imgs.append(fake_img1)
        if self.branch_num > 1:
            h_code2, att1 = \
                self.h_net2(h_code1, c_code, word_embs, mask)
            fake_img2 = self.img_net2(h_code2)
            fake_imgs.append(fake_img2)
            if att1 is not None:
                att_maps.append(att1)
        if self.branch_num > 2:
            h_code3, att2 = \
                self.h_net3(h_code2, c_code, word_embs, mask)
            fake_img3 = self.img_net3(h_code3)
            fake_imgs.append(fake_img3)
            if att2 is not None:
                att_maps.append(att2)

        return fake_imgs, att_maps, mu, logvar


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
    ngf, nef, ncf, branch_num = 32, 256, 100, 3
    z_dim = 100
    print(Generator(ngf, nef, ncf, branch_num, z_dim=z_dim, num_res_block=2, device=torch.device('cpu')))