import numpy as np 
from scores.inception_score import GenImgData, inception_score
from scores.fid_score import calculate_fid_given_paths
from scores.prd_score import prd_score
from text2img_model import Text2ImgModel
from data_utils import BirdsPreprocessor, CaptionTokenizer, BirdsDataset, prepare_data
from torch.utils.data import DataLoader

class Text2ImgTester():
	def __init__(self, data_path, batch_size, embd_size, text_enc_emb_size, pretrained_text_enc, pretrained_image_enc, pretrained_generator, branch_num, is_bert, base_size, device, use_sagan):
		print ("data path: ", data_path)
		self.dataset = self.build_dataset(data_path, base_size)
		print ("self dataset: ", self.dataset)
		self.batch_size = batch_size
		self.data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)


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

		print ("path to data: ", path_to_data)
		preproc = BirdsPreprocessor(data_path=path_to_data, dataset_name='cub')
		tokenizer = CaptionTokenizer(word_to_idx=preproc.word_to_idx, idx_to_word=preproc.idx_to_word)
		dataset = BirdsDataset(mode='test', tokenizer=tokenizer, preprocessor=preproc, branch_num=3, base_size=base_size)

		return dataset

	def get_inception_score(path_to_generated_imgs):
		gen_img_iterator = GenImgData(gen_save_folder)
		mean_val, std_val = inception_score(gen_img_iterator, cuda=False, batch_size=32, resize=False, splits=1)

		return mean_val, std_val

		#def get_fid_score(path_to_generated_imgs, path_to_real_imgs):
		#	pass

if __name__ == '__main__':
	
	model = Text2ImgTester(
			data_path='dataset/CUB_200_2011',
			batch_size = 32,
			embd_size = 256,
			text_enc_emb_size=128,
			pretrained_text_enc='',
			pretrained_image_enc='',
			pretrained_generator='',
			branch_num=3,
			is_bert=False,
			base_size=64,
			device='cpu',
			use_sagan=False)


