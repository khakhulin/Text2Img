import numpy as np
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
from PIL import Image

import torch
from torchvision import transforms

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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
        #img = img.view(img.numel())
        return img

    def __len__(self):
        return self.ds_len


def bin_counts (real_dataset, gen_dataset, number_of_bins=25, batch_size=None):

    if batch_size is None:
        ref_dataloader = torch.utils.data.DataLoader(real_dataset, batch_size=len(real_dataset))
        gen_dataloader = torch.utils.data.DataLoader(gen_dataset, batch_size=len(gen_dataset))
    else:
        ref_dataloader = torch.utils.data.DataLoader(real_dataset, batch_size=batch_size)
        gen_dataloader = torch.utils.data.DataLoader(gen_dataset, batch_size=batch_size)

    real_data = next(iter(ref_dataloader))
    generated_data = next(iter(gen_dataloader))

    real_data = real_data.view(real_data.size(0), int(real_data.numel()/real_data.size(0)))
    generated_data = generated_data.view(generated_data.size(0), int(generated_data.numel() / generated_data.size(0)))

    # binirize real and generated data, plot histogram and found density function
    cluster_data = np.vstack([real_data, generated_data])
    kmeans = MiniBatchKMeans(n_clusters=number_of_bins, n_init=10)
    labels = kmeans.fit(cluster_data).labels_

    eval_labels = labels[:len(real_data)]
    ref_labels = labels[len(real_data):]

    real_density = np.histogram(eval_labels, bins=number_of_bins,
                             range=[0, number_of_bins], density=True)[0]
    gen_density = np.histogram(ref_labels, bins=number_of_bins,
                            range=[0, number_of_bins], density=True)[0]
    return real_density, gen_density


def count_precision_recall(gen_density, real_density, num_angles=1000):
    assert real_density.shape == gen_density.shape

    angles = np.linspace(1e-6, np.pi / 2 - 1e-6, num=num_angles)

    slopes = np.tan(angles)

    # Broadcast slopes so that second dimension will be states of the distribution
    slopes_2d = np.expand_dims(slopes, 1)

    # Broadcast distributions so that first dimension represents the angles
    ref_dist_2d = np.expand_dims(real_density, 0)
    eval_dist_2d = np.expand_dims(gen_density, 0)

    # Compute precision and recall for all angles in one step via broadcasting
    precision = np.minimum(ref_dist_2d * slopes_2d, eval_dist_2d).sum(axis=1)
    recall = precision / slopes

    # handle numerical instabilities leaing to precision/recall just above 1
    max_val = max(np.max(precision), np.max(recall))
    if max_val > 1.001:
        raise ValueError('Detected value > 1.001, this should not happen.')
    precision = np.clip(precision, 0, 1)
    recall = np.clip(recall, 0, 1)

    return precision, recall


def prd_score(real_dataset, gen_dataset, repeat_number=10):
    real_density, generated_density = bin_counts(real_dataset, gen_dataset)

    vectors = [count_precision_recall(real_density, generated_density) for _ in range(repeat_number)]
    vectors = np.array(vectors).mean(axis=0)
    return vectors


def plot_precision_recall(precision, recall, path_to_save, label=''):
    """
    Plots precision recall curve for distribution.

    """

    plt.figure(figsize=(12, 8))
    plt.plot(precision, recall, label=label)
    plt.legend()
    plt.savefig(path_to_save)


def get_plot_as_numpy(precision, recall, label=''):
    """
    Returns precision recall curve as numpy array

    """
    plt.ioff()

    fig = plt.figure(figsize=(12, 8))
    fig.add_subplot(111)
    fig.tight_layout(pad=0)
    plt.plot(precision, recall, labell=label)
    fig.canvas.draw()

    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='', type=str)
    parser.add_argument('--gen_dataset_path', default='', type=str)
    parser.add_argument('--label', default='', type=str)
    parser.add_argument('--path_to_save', default='', type=str)
    args = parser.parse_args()

    ref_dataset = ImageDataset(args.dataset_path)
    gen_dataset = ImageDataset(args.gen_dataset_path)

    print(len(gen_dataset))

    print('Calculate precisions and recalls...')
    (precision, recall) = prd_score(ref_dataset, gen_dataset)

    plot_precision_recall(precision, recall, args.label, args.path_to_save)

