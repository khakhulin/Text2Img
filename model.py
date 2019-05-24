import torch
import torch.nn as nn
from DAMSM import ImageEncoder, TextEncoder
from models import (
    Discriminator64,
    Discriminator128,
    Discriminator256,
    Generator
)
from utils import init_weight


class SuperModel(nn.Module):
    def __init__(
            self,
            embedding_dim,
            n_tokens,
            pretrained_text_encoder_path,
            pretrained_generator_path,
            branch_num,
            cfg_train_b_net_d,
            cfg_gan_b_dcgan,
            cfg_gan_gf_dim,
            cfg_gan_df_dim,
            cfg_gan_z_dim,
            cfg_gan_condition_dim,
            device
    ):
        super(SuperModel, self).__init__()

        if pretrained_text_encoder_path == '':
            print('Error: no pretrained text-image encoders')
            # return

        self.image_encoder = ImageEncoder(multimodal_feat_size=embedding_dim)
        image_encoder_path = pretrained_text_encoder_path.replace('text_encoder', 'image_encoder')
        state_dict = \
            torch.load(image_encoder_path, map_location=lambda storage, loc: storage)
        self.image_encoder.load_state_dict(state_dict)
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', image_encoder_path)
        self.image_encoder.eval()

        self.text_encoder = \
            TextEncoder(n_tokens=n_tokens, emb_size=embedding_dim)
        state_dict = \
            torch.load(pretrained_text_encoder_path,
                       map_location=lambda storage, loc: storage)
        self.text_encoder.load_state_dict(state_dict)
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', pretrained_text_encoder_path)
        self.text_encoder.eval()

        self.discriminators = []
        if cfg_gan_b_dcgan:
            raise
            # ######### WTF IS G_DCGAN????? #############
            # if cfg_tree_branch_num == 1:
            #     D_NET = Discriminator64
            # elif cfg_tree_branch_num == 2:
            #     D_NET = Discriminator128
            # else:  # cfg.TREE.BRANCH_NUM == 3:
            #     D_NET = Discriminator256

            # netG = None
            # self.netsD = None
        else:
            self.generator = Generator(
                ngf=cfg_gan_gf_dim,
                nef=embedding_dim,
                ncf=cfg_gan_condition_dim,
                branch_num=branch_num,
                device=device,
                z_dim=cfg_gan_z_dim
            )
            if branch_num > 0:
                self.discriminators.append(Discriminator64(
                    dim=cfg_gan_df_dim,
                    embd_dim=cfg_gan_gf_dim,
                    # condition= WTF????
                ))
            if branch_num > 1:
                self.discriminators.append(Discriminator128(
                    ndf=cfg_gan_df_dim,
                    embd_dim=cfg_gan_gf_dim,
                ))
            if branch_num > 2:
                self.discriminators.append(Discriminator256(
                    ndf=cfg_gan_df_dim,
                    embd_dim=cfg_gan_gf_dim,
                ))
        self.generator.apply(init_weight)
        print(self.generator)
        for i in range(len(self.discriminators)):
            self.discriminators[i].apply(init_weight)
        print('# of self.netsD', len(self.discriminators))
        self.epoch = 0
        if pretrained_generator_path != '':
            state_dict = \
                torch.load(pretrained_generator_path, map_location=lambda storage, loc: storage)
            self.generator.load_state_dict(state_dict)
            print('Load G from: ', pretrained_generator_path)
            istart = pretrained_generator_path.rfind('_') + 1
            iend = pretrained_generator_path.rfind('.')
            self.epoch = int(pretrained_generator_path[istart:iend]) + 1
            if cfg_train_b_net_d:  # WTF IS THIS ?????
                Gname = pretrained_generator_path
                for i in range(len(self.discriminators)):
                    s_tmp = Gname[:Gname.rfind('/')]
                    Dname = '%s/netD%d.pth' % (s_tmp, i)
                    print('Load D from: ', Dname)
                    state_dict = \
                        torch.load(Dname, map_location=lambda storage, loc: storage)
                    self.discriminators[i].load_state_dict(state_dict)

    def forward(self, images, captions, cap_lens, noise, class_ids, keys):
        batch_size = images.shape[0]
        hidden = self.text_encoder.init_hidden(batch_size)

        words_embeddings, sentence_embedding = \
            self.text_encoder(captions, cap_lens, hidden)
        words_embeddings, sentence_embedding = \
            words_embeddings.detach(), sentence_embedding.detach()
        mask = (captions == 0)
        num_words = words_embeddings.size(2)
        if mask.size(1) > num_words:
            mask = mask[:, :num_words]

        fake_images, attention_maps, mu, logvar = self.generator(
            noise,
            sentence_embedding,
            words_embeddings,
            mask
        )
        return fake_images, mu, logvar


if __name__ == '__main__':
    DEV = torch.device('cuda:2')
    SuperModel(
        embedding_dim=256,
        n_tokens=20,
        pretrained_text_encoder_path='',
        pretrained_generator_path='',
        branch_num=3,
        cfg_train_b_net_d=False,  # ?????
        cfg_gan_b_dcgan=False,    # ?????
        cfg_gan_gf_dim=32,
        cfg_gan_df_dim=64,
        cfg_gan_z_dim=100,
        cfg_gan_condition_dim=64,
        device=DEV
    )
