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
import tqdm
import datetime

from text2img_model import Text2ImgModel
from utils import *
from logger import Logger


class Text2ImgTrainer:
    def __init__(self, batch_size=20, data_path='datasets/CUB_200_2011', continue_from=None,
                 device=torch.device('cuda:2'), args=None):
        self.device = device
        self.batch_size = batch_size
        self.args = args #  TODO find better way to use arguments
        self.dataset = self.build_dataset(data_path)
        self.data_loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size, drop_last=True
        )
        self.loss_logger = Logger(['Epoch'] +
            ['D_loss_%d' % (i) for i in range(args.branch_num)] +
            ['G_loss_%d' % (i) for i in range(args.branch_num)] +
            ['S_loss', 'W_loss', 'KL_loss']
        )
        self.path_to_data = data_path
        self.is_bert = self.args.is_bert
        self.model = self.build_model(
            embedding_dim=args.embd_size,
            n_tokens=self.dataset.n_tokens,
            text_encoder_embd_size=args.text_enc_emb_size, # not used in bert
            pretrained_text_encoder_path='',
            pretrained_image_encoder_path='',
            pretrained_generator_path='',
            branch_num=args.branch_num,
            num_generator_filters=32,
            num_discriminator_filters=64,
            z_dim=100,
            condition_dim=128,
            is_bert_encoder=self.is_bert,
            device=self.device
        )
        
        self.start_epoch = 1
        if not continue_from is None and os.path.exists(continue_from):
            print('Start from checkpoint')
            self.start_epoch = self.model.load_model_ckpt(continue_from)

        self.generator_optimizer, self.discriminator_optimizers = \
            self.build_optimizers(
                model=self.model,
                generator_lr=args.generator_lr,
                discriminator_lr=args.discriminator_lr
            )

        self.avg_snapshot_generator = copy_params(self.model.generator)

    @staticmethod
    def build_model(**kwargs):
        return Text2ImgModel(**kwargs)

    @staticmethod
    def build_dataset(path_to_data):
        preproc = BirdsPreprocessor(data_path=path_to_data, dataset_name='cub')
        tokenizer = CaptionTokenizer(word_to_idx=preproc.word_to_idx, idx_to_word=preproc.idx_to_word)
        dataset = BirdsDataset(tokenizer=tokenizer, preprocessor=preproc, branch_num=args.branch_num)
        image, _, _ = dataset[0]
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

    def train(self, run_name, epochs, log_each, save_img_each, snapshot_each):
        log_dir = os.path.join('trained_models', run_name)
        save_dir = os.path.join('trained_models', run_name)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

        fixed_noise = torch.FloatTensor(
            self.batch_size,
            self.model.z_dim,
        ).normal_(0, 1).to(self.device)
        real_labels = torch.FloatTensor(self.batch_size).fill_(1).to(self.device)
        fake_labels = torch.FloatTensor(self.batch_size).fill_(0).to(self.device)

        batch_passed = 0
        gen_iterations = 0
        
        for epoch in range(self.start_epoch, self.start_epoch + epochs):
            step = 0
            for data in tqdm.tqdm(self.data_loader, total=len(self.data_loader)):
                set_requires_grad_value(self.model.discriminators, True)

                images, captions, cap_lens, masks = prepare_data(data, self.device)

                # batch size can be smaller in the end of the epoch
                noise = torch.FloatTensor(
                    captions.size(0),
                    self.model.z_dim
                ).to(self.device).normal_(0, 1)

                fake_images, mu, logvar, sentence_embedding, words_embeddings = \
                    self.model(
                        captions,
                        cap_lens,
                        noise,
                        masks
                    )

                errD_total = 0
                D_losses = []
                for i in range(len(self.model.discriminators)):
                    self.model.discriminators[i].zero_grad()
                    d_loss = discriminator_loss(
                        self.model.discriminators[i],
                        images[i],
                        fake_images[i],
                        sentence_embedding,
                        real_labels,
                        fake_labels
                    )
                    # backward and update parameters
                    d_loss.backward()
                    self.discriminator_optimizers[i].step()
                    errD_total += d_loss
                    D_losses.append(d_loss.item())

                # compute total loss for training G
                step += 1
                gen_iterations += 1

                set_requires_grad_value(self.model.discriminators, False)
                self.generator_optimizer.zero_grad()
                errG_total, G_losses, W_loss, S_loss = \
                    generator_loss(self.model.discriminators,
                                   self.model.image_encoder,
                                   fake_images, real_labels,
                                   words_embeddings, sentence_embedding,
                                   cap_lens, self.args)
                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                # backward and update parameters
                errG_total.backward()
                self.generator_optimizer.step()
                #  Update average parameters of the generator
                for p, avg_p in zip(self.model.generator.parameters(), self.avg_snapshot_generator):
                    avg_p.mul_(0.999).add_(0.001, p.data)
                
                load_params(self.model.generator, self.avg_snapshot_generator)

            # if gen_iterations % 100 == 0:
            # TODO log every k iteration
            if epoch == 1 or epoch % log_each == 0:
                self.loss_logger.write(
                    epoch, *D_losses, *G_losses, W_loss, S_loss,
                    kl_loss.item()
                )
            self.loss_logger.to_csv(os.path.join(log_dir, 'logs.csv'))
            # make a snapshot
            if epoch % snapshot_each == 0:
                self.model.save_model_ckpt(
                    epoch,
                    os.path.join(save_dir, 'weights%03d.pt' % (epoch))
                )
            # save images

            if epoch % save_img_each == 0:
                #  TODO validation and test part with metric by option
                ## EVAL
                self.model.generator.eval()
                fake_imgs, attention_maps, _, _ = self.model.generator(
                    noise,
                    sentence_embedding,
                    words_embeddings,
                    masks
                )
                save_images(fake_images[-1], None, log_dir, 'vgen_imgs')
                self.model.generator.train()


if __name__ == '__main__':
    assert torch.__version__== '1.1.0'
    args = init_config()
    cur_time = datetime.datetime.now().strftime('%d:%m:%Y:%H-%M-%S')
    run_name = os.path.join(args.exp_name, cur_time)
    trainer = Text2ImgTrainer(
        data_path='dataset/CUB_200_2011', batch_size=2,
        #continue_from='trained_models/26:05:2019:00-28-38/weights002.pt',
        device=torch.device('cuda'), args=args)
    trainer.train(run_name, epochs=10, log_each=1, save_img_each=1, snapshot_each=1)