import numpy as np 
from scores.inception_score import GenImgData, inception_score
from scores.fid_score import calculate_fid_given_paths
from scores.prd_score import prd_score
from text2img_model import Text2ImgModel
from data_utils import BirdsPreprocessor, CaptionTokenizer, BirdsDataset, prepare_data
from data_utils import CocoPreprocessor, CocoDataset
from torch.utils.data import DataLoader
import tqdm
import os
import time
import datetime
import torch
from utils import save_images
from arguments import init_config

class Text2ImgTester():
	def __init__(self, data_path, datasets, batch_size, embd_size, text_enc_emb_size, pretrained_text_enc,\
			pretrained_image_enc, pretrained_generator, branch_num, is_bert, base_size, device, use_sagan):
		
		self.dataset = self.build_dataset(data_path, base_size, dataset_type=datasets)
		self.batch_size = batch_size
		self.data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
		self.device = device

		self.model = Text2ImgModel(
			embedding_dim=embd_size,
			n_tokens=self.dataset.n_tokens,
			text_encoder_embd_size=text_enc_emb_size, # not used in bert
			pretrained_text_encoder_path=pretrained_text_enc,
			pretrained_image_encoder_path=pretrained_image_enc,
			pretrained_generator_path=pretrained_generator,
			branch_num=branch_num,
			num_generator_filters=32,
			num_discriminator_filters=64,
			z_dim=100,
			condition_dim=128,
			is_bert_encoder=is_bert,
			base_size=base_size,
			device=device,
			use_sagan=use_sagan)

	def build_dataset(self, path_to_data, base_size, dataset_type='birds'):
		if dataset_type == 'birds':
			preproc = BirdsPreprocessor(data_path=path_to_data, dataset_name='cub')
			self.test_imgs_paths = preproc.get_test_split_imgs_paths()
			tokenizer = CaptionTokenizer(word_to_idx=preproc.word_to_idx, idx_to_word=preproc.idx_to_word)
			dataset = BirdsDataset(mode='test', tokenizer=tokenizer, preprocessor=preproc, branch_num=3, base_size=base_size)
		elif dataset_type == 'coco':
			preproc = CocoPreprocessor(data_path=path_to_data, dataset_name='coco')
			tokenizer = CaptionTokenizer(word_to_idx=preproc.word_to_idx, idx_to_word=preproc.idx_to_word)
			dataset = CocoDataset(mode='test', tokenizer=tokenizer, preprocessor=preproc,
									branch_num=args.branch_num, base_size=base_size)
			self.test_imgs_paths = dataset.get_test_split_imgs_paths()

		image = dataset[0][0]
		assert image[0].size() == torch.Size([3, base_size, base_size])
		return dataset

	def get_inception_score(self, path_to_generated_imgs):
		gen_img_iterator = GenImgData(gen_save_folder)
		mean_val, std_val = inception_score(gen_img_iterator, cuda=False, batch_size=32, resize=False, splits=1)

		return mean_val, std_val

	def get_scores(self):
		cur_time = datetime.datetime.now().strftime('%d:%m:%Y:%H-%M-%S')
		run_name = os.path.join('gen_exp', cur_time)
		save_dir = os.path.join('generated_images', run_name)
		
		self.model.generator.eval()

		gen_iter = 0 

		for data in tqdm.tqdm(self.data_loader, total=len(self.data_loader)):
			images, captions, cap_lens, masks, class_ids = prepare_data(data, self.device)
			noise = torch.FloatTensor(
				captions.size(0),
				self.model.z_dim
				).to(self.device).normal_(0, 1)

			gen_iter += 1
			gen_images, _, _, _, _ = self.model(captions, cap_lens, noise, masks)
			filenames = [str(gen_iter) + str(i) for i in range(gen_images[-1].size(0))]
			img_tensor = save_images(gen_images[-1], filenames, save_dir, '', gen_images[-1].size(3))

		# inception score calculation
		gen_save_folder = os.path.join(save_dir, 'images', 'iter', str(gen_images[-1].size(3)))
		gen_img_iterator = GenImgData(gen_save_folder)
		mean_val, std_val = inception_score(gen_img_iterator, cuda=False, batch_size=32, resize=False, splits=4)
		print ('Inception Score, mean: {0:.3f}, std: {1:.3f}'.format(mean_val, std_val))
		
		#fid calculation
		paths_to_fid = []
		paths_to_fid.append(gen_save_folder)
		paths_to_fid.append(self.test_imgs_paths)
		fid_val = calculate_fid_given_paths(paths_to_fid, batch_size=1, cuda=False, dims=2048)
		print ("FID value: ", fid_val)


if __name__ == '__main__':
	
	args = init_config()
	
	model = Text2ImgTester(
			data_path = args.data_path,
			datasets = args.datasets,
			batch_size = args.batch_size,
			embd_size = args.embd_size,
			text_enc_emb_size=args.text_enc_emb_size,
			pretrained_text_enc=args.pretrained_text_enc,
			pretrained_image_enc=args.pretrained_image_enc,
			pretrained_generator=args.pretrained_generator,
			branch_num=args.branch_num,
			is_bert=args.is_bert,
			base_size=args.base_size,
			device='cpu',
			use_sagan=args.use_sagan)


	if args.continue_from and os.path.exists(args.continue_from):
		print('Start from checkpoint')
		self.start = self.model.load_model_ckpt(args.continue_from)

	model.get_scores()



