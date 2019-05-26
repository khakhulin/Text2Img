import numpy as np
from text2img_model import Text2ImgModel



class TestText2Img():

	def __init__(self, save_gen_img_path, data_path='datasets/CUB_200_2011'):
        
		self.dataset = self.build_dataset(data_path)

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

		self.path_to_data = data_path
		self.gen_img_path = save_gen_img_path


	def test_model(self):
		
		

	@staticmethod
    def build_dataset(path_to_data):
        preproc = BirdsPreprocessor(data_path=path_to_data, dataset_name='cub')
        tokenizer = CaptionTokenizer(word_to_idx=preproc.word_to_idx, idx_to_word=preproc.idx_to_word)
        dataset = BirdsDataset(mode='test', tokenizer=tokenizer, preprocessor=preproc, branch_num=args.branch_num)
        image, _, _ = dataset[0]
        assert image[0].size() == torch.Size([3, 64, 64])
        return dataset


    '''
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
	'''



'''
if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Test Text2Img')
	parser.add_argument('--model_ckpt_path', default='', type=str)
	args = parser.parse_args()

	DEV = torch.device('cpu')

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

	#model.load_model_ckpt(path = model_ckpt_path)
'''

