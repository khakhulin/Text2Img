import torch
import torch.nn as nn
from DAMSM import ImageEncoder, TextEncoder, BertEncoder
from models import (
    Discriminator64,
    Discriminator128,
    Discriminator256,
    Generator
)
from utils import init_weight, copy_params


class Text2ImgModel(nn.Module):
    def __init__(
            self,
            embedding_dim,
            n_tokens,
            text_encoder_embd_size,
            pretrained_text_encoder_path,
            pretrained_image_encoder_path,
            pretrained_generator_path,
            branch_num,
            num_generator_filters,
            num_discriminator_filters,
            z_dim,
            condition_dim,
            is_bert_encoder,
            device
    ):
        super(Text2ImgModel, self).__init__()

        self.z_dim = z_dim
        self.is_bert_encoder = is_bert_encoder
        self.image_encoder = ImageEncoder(
            multimodal_feat_size=embedding_dim
        ).to(device)

        if pretrained_image_encoder_path != '':
            state_dict = torch.load(
                pretrained_image_encoder_path,
                map_location=lambda storage, loc: storage
            )
            self.image_encoder.load_state_dict(state_dict)
            print('Load image encoder from:', pretrained_image_encoder_path)
        else:
            print('Warning: no pretrained image encoder')

        for p in self.image_encoder.parameters():
            p.requires_grad = False

        self.image_encoder.eval()
        if self.is_bert_encoder:
            self.text_encoder = BertEncoder(emb_size=embedding_dim).to(device)
        else:
            self.text_encoder = TextEncoder(
                n_tokens=n_tokens, text_feat_size=embedding_dim,
                emb_size=text_encoder_embd_size
            ).to(device)

        if pretrained_text_encoder_path != '':
            state_dict = \
                torch.load(pretrained_text_encoder_path,
                           map_location=lambda storage, loc: storage)
            self.text_encoder.load_state_dict(state_dict)

            print('Load text encoder from:', pretrained_text_encoder_path)
        else:
            print('Warning: no pretrained text encoder')

        for p in self.text_encoder.parameters():
            p.requires_grad = False

        self.text_encoder.eval()

        self.generator = Generator(
            ngf=num_generator_filters,
            emb_dim=embedding_dim,
            ncf=condition_dim,
            branch_num=branch_num,
            device=device,
            z_dim=z_dim
        ).to(device)
        self.discriminators = []
        if branch_num > 0:
            self.discriminators.append(Discriminator64(
                dim=num_discriminator_filters,
                embd_dim=embedding_dim,
            ).to(device))
        if branch_num > 1:
            self.discriminators.append(Discriminator128(
                ndf=num_discriminator_filters,
                embd_dim=embedding_dim,
            ).to(device))
        if branch_num > 2:
            self.discriminators.append(Discriminator256(
                ndf=num_discriminator_filters,
                embd_dim=embedding_dim,
            ).to(device))
        self.generator.apply(init_weight)

        for i in range(len(self.discriminators)):
            self.discriminators[i].apply(init_weight)

        if pretrained_generator_path != '':
            state_dict = torch.load(
                pretrained_generator_path,
                map_location=lambda storage, loc: storage
            )
            self.generator.load_state_dict(state_dict)
            print('Load generator from: ', pretrained_generator_path)

    def forward(self, captions, cap_lens, noise, masks):
        if not self.is_bert_encoder:
            words_embeddings, sentence_embedding = \
                self.text_encoder(captions, cap_lens)
        else:
            words_embeddings, sentence_embedding = \
                self.text_encoder(captions, cap_lens, masks)
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
        return fake_images, mu, logvar, sentence_embedding, words_embeddings
    
    def save_model_ckpt(self, epoch, path):
        torch.save({
            'epoch': epoch,
            'img_enc': self.image_encoder.state_dict(),
            'txt_enc': self.text_encoder.state_dict(),
            'discriminator': [d.state_dict() for d in self.discriminators],
            'generator': self.generator.state_dict()
        }, path)
    
    def load_model_ckpt(self, path):
        weights = torch.load(path)
        self.image_encoder.load_state_dict(weights['img_enc'])
        self.text_encoder.load_state_dict(weights['txt_enc'])

        for i, d in enumerate(self.discriminators):
            d.load_state_dict(weights['discriminator'][i])
        self.generator.load_state_dict(weights['generator'])
        
        return weights['epoch']


if __name__ == '__main__':
    DEV = torch.device('cuda:2')
    model = Text2ImgModel(
        embedding_dim=256,
        n_tokens=20,
        text_encoder_embd_size=128,
        pretrained_text_encoder_path='',
        pretrained_image_encoder_path='',
        pretrained_generator_path='',
        branch_num=3,
        num_generator_filters=32,
        num_discriminator_filters=64,
        z_dim=100,
        condition_dim=128,  # should be half of embedding_dim
        is_bert_encoder=False,
        device=DEV
    )
    print(model)

