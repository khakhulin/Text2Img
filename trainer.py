import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from arguments import init_config
from data_utils import BirdsPreprocessor, CaptionTokenizer, BirdsDataset, prepare_data
from losses import discriminator_loss, generator_loss, KL_loss
import os
import time
import numpy as np
import sys

from text2img_model import Text2ImgModel
from utils import *


class Text2ImgTrainer:
    def __init__(self, batch_size=20, data_path='datasets/CUB_200_2011', device=torch.device('cuda:2'), args=None):
        self.device = device
        self.batch_size = batch_size
        self.args = args #  TODO find better way to use arguments
        self.dataset = self.build_dataset(data_path)
        self.data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size)
        self.path_to_data = data_path
        self.is_bert = self.args.is_bert
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
            is_bert_encoder=self.is_bert,
            device=self.device
        )

        self.generator_optimizer, self.discriminator_optimizers = \
            self.build_optimizers(
                model=self.model,
                generator_lr=0.1,
                discriminator_lr=0.1
            )

        self.avg_snapshot_generator = copy_params(self.model.generator)

    @staticmethod
    def build_model(**kwargs):
        return Text2ImgModel(**kwargs)

    @staticmethod
    def build_dataset(path_to_data):
        preproc = BirdsPreprocessor(data_path=path_to_data, dataset_name='cub')
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

        batch_passed = 0
        gen_iterations = 0
        for epoch in range(1, epochs):
            start_t = time.time()
            step = 0
            for data in self.data_loader:
                set_requires_grad_value(self.model.discriminators, True)

                images, captions, cap_lens, masks = prepare_data(data, self.device)

                noise.normal_(0, 1)
                fake_images, mu, logvar, sentence_embedding, words_embeddings = \
                    self.model(
                        captions,
                        cap_lens,
                        noise,
                        masks
                    )

                errD_total = 0
                D_logs = ''
                for i in range(len(self.model.discriminators)):
                    self.model.discriminators[i].zero_grad()
                    discirminato_loss = discriminator_loss(
                        self.model.discriminators[i],
                        images[i],
                        fake_images[i],
                        sentence_embedding,
                        real_labels,
                        fake_labels
                    )
                    # backward and update parameters
                    discirminato_loss.backward()
                    self.discriminator_optimizers[i].step()
                    errD_total += discirminato_loss
                    D_logs += 'discr_loss_{0} : {1:.2f} '.format(i, discirminato_loss.item())

                # compute total loss for training G
                step += 1
                gen_iterations += 1

                set_requires_grad_value(self.model.discriminators, False)
                self.generator_optimizer.zero_grad()
                errG_total, G_logs = generator_loss(self.model.discriminators,
                                                    self.model.image_encoder,
                                                    fake_images, real_labels,
                                                    words_embeddings, sentence_embedding,
                                                    cap_lens, self.args)
                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                G_logs += 'kl_loss: {0:.2f} '.format(kl_loss.item())
                # backward and update parameters
                errG_total.backward()
                self.generator_optimizer.step()
                #  Update average parameters of the generator
                for p, avg_p in zip(self.model.generator.parameters(), self.avg_snapshot_generator ):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                # if gen_iterations % 100 == 0:
                print(D_logs + '\n' + G_logs)
                # save images
                exit()
                if gen_iterations % 1000 == 0:
                    load_params(self.model.generator, self.avg_snapshot_generator)
                    #  TODO validation
            end_t = time.time()



if __name__ == '__main__':
    assert torch.__version__== '1.1.0'
    args = init_config()
    trainer = Text2ImgTrainer(data_path='datasets/CUB_200_2011', batch_size=2, device=torch.device('cpu'),args=args)
    trainer.train(epochs=10)
