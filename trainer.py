import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from PIL import Image

# from miscc.config import cfg
# from miscc.utils import mkdir_p
# from miscc.utils import build_super_images, build_super_images2
# from miscc.utils import weights_init, load_params, copy_G_params
# from model import G_DCGAN, G_NET
# from datasets import prepare_data
# from model import RNN_ENCODER, CNN_ENCODER

from data_utils import BirdsPreprocessor, CaptionTokenizer, BirdsDataset
from losses import words_loss
from losses import discriminator_loss, generator_loss, KL_loss
import os
import time
import numpy as np
import sys

from text2img_model import Text2ImgModel


class Text2ImgTrainer:
    def __init__(self):
        self.device = torch.device('cuda:2')
        self.batch_size = 20

        self.dataset = self.build_dataset()
        self.data_loader = \
            torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch_size)

        self.model = self.build_model(
            embedding_dim=256,
            n_tokens=self.dataset.n_tokens,
            pretrained_text_encoder_path='',
            pretrained_image_encoder_path='',
            pretrained_generator_path='',
            branch_num=2,
            num_generator_filters=32,
            num_discriminator_filters=64,
            z_dim=100,
            condition_dim=128,
            device=self.device
        )

        self.generator_optimizer, self.discriminator_optimizers = \
            self.build_optimizers(
                model=self.model,
                generator_lr=0.1,
                discriminator_lr=0.1
            )

        # self.batch_size = cfg.TRAIN.BATCH_SIZE
        # self.max_epoch = cfg.TRAIN.MAX_EPOCH
        # self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        #
        # self.n_words = n_words
        # self.ixtoword = ixtoword
        # self.data_loader = data_loader
        # self.num_batches = len(self.data_loader)

    @staticmethod
    def build_model(**kwargs):
        return Text2ImgModel(**kwargs)

    @staticmethod
    def build_data_loader(batch_size):
        preproc = BirdsPreprocessor(data_path='datasets/CUB_200_2011', dataset_name='cub')
        assert len(preproc.train) == 9813
        assert len(preproc.test) == 1179
        tokenizer = CaptionTokenizer(word_to_idx=preproc.word_to_idx)
        dataset = BirdsDataset(tokenizer=tokenizer, preprocessor=preproc, branch_num=2)
        image, caption, length = dataset[0]
        assert image[0].size() == torch.Size([3, 64, 64])
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)
        return data_loader

    @staticmethod
    def build_dataset():
        preproc = BirdsPreprocessor(data_path='datasets/CUB_200_2011', dataset_name='cub')
        assert len(preproc.train) == 9813
        assert len(preproc.test) == 1179
        tokenizer = CaptionTokenizer(word_to_idx=preproc.word_to_idx)
        dataset = BirdsDataset(tokenizer=tokenizer, preprocessor=preproc, branch_num=2)
        image, caption, length = dataset[0]
        assert image[0].size() == torch.Size([3, 64, 64])
        return dataset

    @staticmethod
    def build_optimizers(
            model: Text2ImgModel,
            generator_lr,
            discriminator_lr
    ):
        discriminator_optimizers = []
        discriminator_count = len(model.discriminators)
        for i in range(discriminator_count):
            opt = optim.Adam(
                model.discriminators[i].parameters(),
                lr=discriminator_lr,
                betas=(0.5, 0.999)
            )
            discriminator_optimizers.append(opt)

        generator_optimizer = optim.Adam(
            model.generator.parameters(),
            lr=generator_lr,
            betas=(0.5, 0.999)
        )

        return generator_optimizer, discriminator_optimizers

    def train(self, epochs):
        noise = torch.FloatTensor(
            self.batch_size,
            self.model.z_dim
        ).to(self.device)
        fixed_noise = torch.FloatTensor(
            self.batch_size,
            self.model.z_dim,
        ).normal_(0, 1).to(self.device)
        real_labels = torch.FloatTensor(self.batch_size).fill_(1).to(self.device)
        fake_labels = torch.FloatTensor(self.batch_size).fill_(0).to(self.device)
        match_labels = torch.LongTensor(range(self.batch_size)).to(self.device)

        batch_passed = 0
        for epoch in range(1, epochs):
            start_t = time.time()
            step = 0
            for data in self.data_loader:
                # reset requires_grad to be trainable for all Ds
                # self.set_requires_grad_value(netsD, True)

                images, captions, cap_lens = data
                # sorted_cap_lens, sorted_cap_indices = \
                #     torch.sort(cap_lens, dim=0, descending=True)
                # sorted_images = []
                # for i in range(len(images)):
                #     sorted_images.append(images[i].to(self.device))
                #
                # captions = captions[sorted_cap_indices].squeeze()
                # images = sorted_images
                # cap_lens = sorted_cap_lens

                noise.normal_(0, 1)
                fake_images, mu, logvar, sentence_embedding = \
                    self.model(
                        images,
                        captions.to(self.device),
                        cap_lens.to(self.device),
                        noise
                    )

                errD_total = 0
                D_logs = ''
                for i in range(len(self.model.discriminators)):
                    self.model.discriminators[i].zero_grad()
                    errD = discriminator_loss(
                        self.model.discriminators[i],
                        images[i],
                        fake_images[i],
                        sentence_embedding,
                        real_labels,
                        fake_labels
                    )
                    # backward and update parameters
                    errD.backward()
                    optimizersD[i].step()
                    errD_total += errD
                    D_logs += 'errD%d: %.2f ' % (i, errD.data[0])

                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G
                step += 1
                gen_iterations += 1

                # do not need to compute gradient for Ds
                # self.set_requires_grad_value(netsD, False)
                netG.zero_grad()
                errG_total, G_logs = \
                    generator_loss(netsD, image_encoder, fake_imgs, real_labels,
                                   words_embs, sent_emb, match_labels, cap_lens, class_ids)
                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                G_logs += 'kl_loss: %.2f ' % kl_loss.data[0]
                # backward and update parameters
                errG_total.backward()
                optimizerG.step()
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_iterations % 100 == 0:
                    print(D_logs + '\n' + G_logs)
                # save images
                if gen_iterations % 1000 == 0:
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    self.save_img_results(netG, fixed_noise, sent_emb,
                                          words_embs, mask, image_encoder,
                                          captions, cap_lens, epoch, name='average')
                    load_params(netG, backup_para)
                    #
                    # self.save_img_results(netG, fixed_noise, sent_emb,
                    #                       words_embs, mask, image_encoder,
                    #                       captions, cap_lens,
                    #                       epoch, name='current')
            end_t = time.time()

            print('''[%d/%d][%d]
                          Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total.data[0], errG_total.data[0],
                     end_t - start_t))

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                self.save_model(netG, avg_param_G, netsD, epoch)

        self.save_model(netG, avg_param_G, netsD, self.max_epoch)


if __name__ == '__main__':
    print(torch.__version__)
    trainer = Text2ImgTrainer()
    trainer.train(epochs=10)
