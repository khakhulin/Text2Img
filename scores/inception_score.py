import numpy as np 
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

from scipy.stats import entropy
from tqdm import tqdm


class ImageList(torch.utils.data.Dataset):
    def __init__(self, img_list):
        self.img_list = img_list

    def __getitem__(self, index):
        return self.img_list[index]

    def __len__(self):
        return len(self.img_list)

# https://github.com/sbarratt/inception-score-pytorch
def inception_score(imgs, cuda = False, batch_size=32, resize=False, splits=1):
	
	N = len(imgs)

	assert batch_size > 0
	assert N > batch_size

	if cuda:
		dtype = torch.cuda.FloatTensor
	else:
		dtype = torch.FloatTensor

	dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

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

#CIFAR Example 

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
