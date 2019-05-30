import datetime

import numpy as np
import torch.optim as optim
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from arguments import init_config
from data_utils import BirdsPreprocessor, CaptionTokenizer, BirdsDataset, prepare_data
from data_utils import CocoPreprocessor, CocoDataset
from modules.losses import discriminator_loss, generator_loss, KL_loss
import os
import time
import numpy as np
import sys
import tqdm
import datetime

from logger import Logger
from modules.losses import discriminator_loss, generator_loss, KL_loss
from scores.inception_score import GenImgData
from scores.inception_score import inception_score
from text2img_model import Text2ImgModel
from utils import *


class Text2ImgTrainer:
    def __init__(self, args=None):
        if args.cuda and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{args.cuda_device}')
        else:
            self.device = 'cpu'
        self.writer = SummaryWriter()
        self.batch_size = args.batch_size
        self.args = args #  TODO find better way to use arguments
        self.dataset = self.build_dataset(
            args.data_path, args.base_size,
            dataset_type=args.dataset_type
        )
        self.data_loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=True, drop_last=True
        )
        self.loss_logger = Logger(['Epoch'] +
            ['D_loss_%d' % (i) for i in range(args.branch_num)] +
            ['G_loss_%d' % (i) for i in range(args.branch_num)] +
            ['S_loss', 'W_loss', 'KL_loss']
        )
        self.path_to_data = args.data_path
        self.is_bert = self.args.is_bert
        self.use_sagan = self.args.use_sagan
        self.avg_snapshot_generator = None

        self.model = self.build_model(
            embedding_dim=args.embd_size,
            n_tokens=self.dataset.n_tokens,
            text_encoder_embd_size=args.text_enc_emb_size, # not used in bert
            pretrained_text_encoder_path=args.pretrained_text_enc,
            pretrained_image_encoder_path=args.pretrained_image_enc,
            pretrained_generator_path=args.pretrained_generator,
            branch_num=args.branch_num,
            num_generator_filters=32,
            num_discriminator_filters=64,
            z_dim=100,
            condition_dim=128,
            is_bert_encoder=self.is_bert,
            base_size=args.base_size,
            device=self.device,
            use_sagan=self.use_sagan
        )
        
        self.start = 0

        # Load average weights from the specified location
        if args.use_average_weights:
            if args.average_weights:
                print('Load average state of generator')
                back_up = copy_params(self.model.generator)
                self.model.generator.load_state_dict(
                    torch.load(args.average_weights)
                )
                self.avg_snapshot_generator = copy_params(self.model.generator)
                load_params(self.model.generator, back_up)

        if args.continue_from and os.path.exists(args.continue_from):
            print('Start from checkpoint')
            # Load model checkpoint
            self.start = self.model.load_model_ckpt(args.continue_from)

        self.generator_optimizer, self.discriminator_optimizers = \
            self.build_optimizers(
                model=self.model,
                generator_lr=args.generator_lr,
                discriminator_lr=args.discriminator_lr
            )
        # Accumulate from the current generator state if no path given
        if args.use_average_weights:
            if self.avg_snapshot_generator is None:
                self.avg_snapshot_generator = copy_params(self.model.generator)

    @staticmethod
    def build_model(**kwargs):
        return Text2ImgModel(**kwargs)

    @staticmethod
    def build_dataset(path_to_data, base_size, dataset_type='birds'):
        if dataset_type == 'birds':
            preproc = BirdsPreprocessor(data_path=path_to_data, dataset_name='cub')
            tokenizer = CaptionTokenizer(word_to_idx=preproc.word_to_idx, idx_to_word=preproc.idx_to_word)
            dataset = BirdsDataset(mode='train', tokenizer=tokenizer, preprocessor=preproc,
                                   branch_num=args.branch_num, base_size=base_size)
        elif dataset_type == 'coco':
            preproc = CocoPreprocessor(data_path=path_to_data, dataset_name='coco')
            tokenizer = CaptionTokenizer(word_to_idx=preproc.word_to_idx, idx_to_word=preproc.idx_to_word)
            dataset = CocoDataset(mode='train', tokenizer=tokenizer, preprocessor=preproc,
                                  branch_num=args.branch_num, base_size=base_size)

        image = dataset[0][0]
        assert image[0].size() == torch.Size([3, base_size, base_size])
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

    def train(self, run_name, epochs, n_images=5):
        log_dir = os.path.join('trained_models', run_name)
        save_dir = os.path.join('trained_models', run_name)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        # Sample captions for image generation
        val_dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=n_images
        )
        val_data = next(iter(val_dataloader))
        del val_dataloader
        val_img, val_cap, val_cap_len, val_mask, _ = prepare_data(
            val_data, self.device
        )
        val_cap_str = self.dataset.tensor_to_caption(val_cap)
        val_cap_str = ['%d. '%(i)+cap_str for i, cap_str in enumerate(val_cap_str)]
        self.writer.add_text('captions', "\n".join(val_cap_str))

        # fixed_noise = torch.FloatTensor(
        #     self.batch_size,
        #     self.model.z_dim,
        # ).normal_(0, 1).to(self.device)
        real_labels = torch.FloatTensor(self.batch_size).fill_(1).to(self.device)
        fake_labels = torch.FloatTensor(self.batch_size).fill_(0).to(self.device)

        # batch_passed = 0
        gen_iterations = self.start
        D_losses = np.zeros((len(self.model.discriminators),))
        G_losses = np.zeros((len(self.model.discriminators),))
        W_loss = 0.0
        S_loss = 0.0
        KLD_loss = 0.0
        
        for epoch in range(epochs):
            print('Epoch %03d' % (epoch))
            # step = 0

            for data in tqdm.tqdm(self.data_loader, total=len(self.data_loader)):
                set_requires_grad_value(self.model.discriminators, True)

                images, captions, cap_lens, masks, class_ids = \
                    prepare_data(data, self.device)

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

                for i in range(len(self.model.discriminators)):
                    self.model.discriminators[i].zero_grad()
                    d_loss = discriminator_loss(
                        self.model.discriminators[i],
                        images[i] + torch.randn_like(images[i]),
                        fake_images[i] + torch.randn_like(images[i]),
                        sentence_embedding,
                        real_labels,
                        fake_labels,
                        loss_type=args.loss_type
                    )
                    # backward and update parameters
                    d_loss.backward()
                    self.discriminator_optimizers[i].step()
                    errD_total += d_loss
                    D_losses[i] += d_loss.item()

                # compute total loss for training G
                # step += 1
                gen_iterations += 1

                set_requires_grad_value(self.model.discriminators, False)
                self.generator_optimizer.zero_grad()
                errG_total, g_losses, w_loss, s_loss = \
                    generator_loss(
                        self.model.discriminators,
                        self.model.image_encoder,
                        fake_images, real_labels,
                        words_embeddings, sentence_embedding,
                        cap_lens, self.args, class_ids=class_ids
                    )

                assert len(G_losses) == len(g_losses), 'generator loss error'
                for i in range(len(G_losses)):
                    G_losses[i] += g_losses[i]
                W_loss += w_loss
                S_loss += s_loss

                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                KLD_loss += kl_loss.item()
                # backward and update parameters
                errG_total.backward()
                self.generator_optimizer.step()
                #  Update average parameters of the generator
                if self.args.use_average_weights:
                    for p, avg_p in zip(self.model.generator.parameters(),
                                        self.avg_snapshot_generator):
                        avg_p.mul_(0.999).add_(0.001, p.data)

                # Track generator gradients
                top, bottom = get_top_bottom_mean_grad(
                    self.model.generator.parameters()
                )
                self.writer.add_scalars(
                    'grad/G', {'top': top, 'bottom': bottom}, gen_iterations
                )
                # Track discriminators gradients
                for i in range(len(self.model.discriminators)):
                    top, bottom = get_top_bottom_mean_grad(
                        self.model.discriminators[i].parameters()
                    )
                    self.writer.add_scalars(
                        'grad/D%d'%(i),
                        {'top': top, 'bottom': bottom}, gen_iterations
                    )

                if gen_iterations % args.log_every == 0:
                    # Mean losses
                    D_losses /= args.log_every
                    G_losses /= args.log_every
                    W_loss /= args.log_every
                    S_loss /= args.log_every
                    KLD_loss /= args.log_every
                    # Save losses to log file
                    self.loss_logger.write(
                        epoch, *D_losses, *G_losses,
                        W_loss, S_loss, KLD_loss
                    )
                    self.loss_logger.to_csv(os.path.join(log_dir, 'logs.csv'))
                    # Save losses to tensorboard
                    self.writer.add_scalars(
                        'losses/G_losses',
                        {'g%d'%(i): val for i, val in enumerate(G_losses)},
                        gen_iterations
                    )
                    self.writer.add_scalars(
                        'losses/D_losses',
                        {'d%d'%(i): val for i, val in enumerate(D_losses)},
                        gen_iterations
                    )
                    self.writer.add_scalars(
                        'losses/SW_losses',
                        {'s_loss': S_loss, 'w_loss': W_loss},
                        gen_iterations
                    )
                    self.writer.add_scalar(
                        'losses/KL_loss', KLD_loss, gen_iterations
                    )
                    # Erase accumulated losses
                    D_losses.fill(0)
                    G_losses.fill(0)
                    W_loss = 0.0
                    S_loss = 0.0
                    KLD_loss = 0.0
                # save images
                if gen_iterations % args.log_every == 0:
                    #  TODO validation and test part with metric by option
                    ## EVAL
                    self.model.generator.eval()

                    if self.args.use_average_weights:
                        backup_params = copy_params(self.model.generator)
                        load_params(self.model.generator, self.avg_snapshot_generator)

                    val_noise = torch.FloatTensor(
                        val_cap.size(0),
                        self.model.z_dim
                    ).to(self.device).normal_(0, 1)

                    with torch.no_grad():
                        gen_imgs_stack, _, _, _, _ = \
                            self.model(
                                val_cap,
                                val_cap_len,
                                val_noise,
                                val_mask
                            )
                    
                    if self.args.use_average_weights:
                        load_params(self.model.generator, backup_params)

                    for i, gen_imgs in enumerate(gen_imgs_stack):
                        size = 64*(2**i)
                        img_tensor = save_images(gen_imgs, None, log_dir, gen_iterations, size)
                        img_tensor = make_grid(img_tensor, nrow=n_images, padding=5)
                        self.writer.add_image(
                            'images/%d' % (size),
                            img_tensor, gen_iterations
                        )

                    #val_fid_score = fid_score(gen_imgs, val_img, batch_size=4, cuda=self.device, dims=2048)
                    #self.writer.add_scalar('metrics/fid', val_fid_score, epoch)

                    #precision, recall = prd_score(val_img, gen_imgs)
                    #pr_plot = get_plot_as_numpy(precision, recall)
                    #self.writer.add_image('metrics/prd_score', pr_plot, epoch)

                    self.model.generator.train()
                # make a snapshot
                if gen_iterations % args.snapshot_every == 0:
                    self.model.save_model_ckpt(
                        gen_iterations,
                        os.path.join(save_dir, 'weights%05d.pt' % (gen_iterations))
                    )
                    if self.args.use_average_weights:
                        backup_params = copy_params(self.model.generator)
                        load_params(self.model.generator, self.avg_snapshot_generator)

                        torch.save(
                            self.model.generator.state_dict(),
                            os.path.join(save_dir, 'avg_weights%05d.pt' % (gen_iterations))
                        )

                        load_params(self.model.generator, backup_params)

            if (args.inception_score_on_validation):
                gen_save_folder = os.path.join(log_dir, 'images', 'iter'+str(gen_iterations), str(256))
                gen_img_iterator = GenImgData(gen_save_folder)
                val_inception_score = inception_score(gen_img_iterator, batch_size=1)
                self.writer.add_scalar('metrics/inception', val_inception_score[0], epoch)

        self.writer.close()

if __name__ == '__main__':
    assert torch.__version__== '1.1.0'
    args = init_config()
    cur_time = datetime.datetime.now().strftime('%d:%m:%Y:%H-%M-%S')
    run_name = os.path.join(args.exp_name, cur_time)
    trainer = Text2ImgTrainer(args=args)
    trainer.train(run_name, epochs=args.max_epoch)