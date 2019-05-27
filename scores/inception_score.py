import numpy as np 
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import argparse

from torchvision.models.inception import inception_v3

from scipy.stats import entropy
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def collate(batch):
	elem_type = type(batch[0])
	if isinstance(batch[0], torch.Tensor):
		out = None
		numel = sum([x.numel() for x in batch])
		storage = batch[0].storage()._new_shared(numel)
		out = batch[0].new(storage)
		return torch.stack(batch, 0, out=out)

class ImageDataset(torch.utils.data.Dataset):
	def __init__(self, path_to_imgs):
		self.path_to_imgs = path_to_imgs

		self.img_path_list = []
		for img_dir in os.listdir(self.path_to_imgs):
			cur_subdir = os.path.join(self.path_to_imgs, img_dir)
			if (os.path.isdir(cur_subdir)):
				for img in os.listdir(cur_subdir):
					if img.endswith(".jpg"):
						self.img_path_list.append(os.path.join(cur_subdir, img))

		self.preprocess()

	#NOTE: Do preprocessing to remove 1xHxW (1 channel) images
	def preprocess(self):
		idx = 0
		print ("preprocessing ...")
		for img_path in tqdm(self.img_path_list):
			img = Image.open(img_path)
			img = transforms.ToTensor()(img)
			if (img.size(0) != 3):
				del self.img_path_list[idx]
			idx += 1

		self.ds_len = len(self.img_path_list)

	def __getitem__(self, index):
		img = Image.open(self.img_path_list[index])
		img = transforms.Resize((299, 299))(img)
		img = transforms.ToTensor()(img)
		img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
		return img

	def __len__(self):
		return self.ds_len


def inception_score(imgs, cuda = False, batch_size=32, resize=False, splits=10):
	
	N = len(imgs)

	assert batch_size > 0
	assert N > batch_size

	if cuda:
		dtype = torch.cuda.FloatTensor
	else:
		dtype = torch.FloatTensor

	dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size, collate_fn=collate)

	inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
	inception_model.eval();

	up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).type(dtype)

	def get_pred(x):
		if resize:
			x = up(x)
		x = inception_model(x)
		return F.softmax(x, dim=1).data.cpu().numpy()

	preds = np.zeros((N, 1000))

	for i, batch in enumerate(tqdm(dataloader), 0):
		batch = batch.type(dtype)
		batchv = Variable(batch)
		batch_size_i = batch.size()[0]

		preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batchv)

	split_scores = []

	for k in tqdm(range(splits)):
		
		part = preds[k * (N // splits): (k+1) * (N // splits), :]
		py = np.mean(part, axis=0)
		scores = []
		
		for i in range(part.shape[0]):
			pyx = part[i, :]
			#Since second param of entropy() is not None, 
			#entropy() gives Kullback-Leibler divergence
			scores.append(entropy(pyx, py))
		
		split_scores.append(np.exp(np.mean(scores)))

	return np.mean(split_scores), np.std(split_scores)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--dataset_path', default='', type=str)
	args = parser.parse_args()

	img_dataset = ImageDataset(args.dataset_path)

	print ("Bird dataset len: ", img_dataset.__len__())

	print ("Calculating Inception Score...")
	print (inception_score(img_dataset, cuda=False, batch_size=32, resize=True, splits=10))


#CIFAR Example
'''
if __name__ == '__main__':
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
        	#HACK: fast testing
            return (len(self.orig) - 49000)

    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    cifar = dset.CIFAR10(root='data/', download=True,
                             transform=transforms.Compose([
                                 transforms.Scale(32),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])
    )

    IgnoreLabelDataset(cifar)

    print ("Calculating Inception Score...")
    print (inception_score(IgnoreLabelDataset(cifar), cuda=False, batch_size=32, resize=True, splits=1))
'''
