from abc import ABC, abstractmethod

import os
import pickle

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.coco import CocoCaptions
try:
    from pycocotools.coco import COCO
except ImportError:
    pass

try:
    from pytorch_pretrained_bert import BertTokenizer
except ImportError:
    pass

from nltk.tokenize import RegexpTokenizer

MAX_SEQ_LEN = 30

def get_preprocessor(dataset_name, data_dir):
    if dataset_name == "cub":
        return BirdsPreprocessor(dataset_name, data_dir)


class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def train_test_split(self):
        pass


class BirdsPreprocessor(DataPreprocessor):
    def __init__(self, dataset_name, data_path):
        super(BirdsPreprocessor, self).__init__(data_path=data_path)

        self.data_dir = data_path
        self.images_path_file = os.path.join(data_path, "images.txt")
        self.data_path = os.path.join(data_path, "images")
        self.captions_path = os.path.join(data_path, "text_c10")
        self.processed_data = "data/"

        with open(self.images_path_file, "r") as imf:
            self.file_names = imf.read().splitlines()

        self.vocab_path = self.processed_data + dataset_name + "_vocab.pkl"
        self.train_test_split_path = self.processed_data + dataset_name + "_data.pkl"

        if not os.path.exists(self.processed_data):
            os.makedirs(self.processed_data)

        self.vocabs = {"idx_to_word": {}, "word_to_idx": {}}

        if os.path.exists(self.vocab_path):
            with open(self.vocab_path, "rb") as bow_file:
                self.vocabs = pickle.load(bow_file)
        else:
            self.vocabs = self.preprocess()

        self.idx_to_word = self.vocabs["idx_to_word"]
        self.word_to_idx = self.vocabs["word_to_idx"]

        self.splitted_data = {"train": None, "val": None, "test": None}

        if os.path.exists(self.train_test_split_path):
            with open(self.train_test_split_path, "rb") as tt_file:
                self.splitted_data = pickle.load(tt_file)
        else:
            self.splitted_data = self.train_test_split()

        self.train = self.splitted_data["train"]
        self.test = self.splitted_data["test"]
        self.val = self.splitted_data["val"]

    def get_test_split_imgs_paths(self):
        test_img_list = [os.path.join(self.data_dir, "images", path) for path in self.test]
        return test_img_list

    def preprocess(self):
        """
        create vocabulary, tokenize captions with len>0
        :return: vocab dict
        """
        all_captions =[]
        for img in self.file_names:
            name_parts = img.split()
            file_name = name_parts[1]
            txt_name = '.'.join(file_name.split('.')[0:-1]) + '.txt'
            txt_path = os.path.join(self.captions_path, txt_name)
            with open(txt_path, 'r', encoding='utf-8') as txt_file:
                captions = txt_file.read().splitlines()

            for cap in captions:
                if len(cap) == 0:
                    continue
                cap = cap.replace(u"\ufffd\ufffd", u" ")

                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(cap.lower())

                if len(tokens) == 0:
                    print('cap', cap)
                    continue

                tokens_new = []
                for t in tokens:
                    t = t.encode('ascii', 'ignore').decode('ascii')
                    if len(t) > 0:
                        tokens_new.append(t)
                all_captions.extend(tokens_new)

        vocab = np.unique(all_captions)

        idx_to_word = dict()
        idx_to_word[0] = '<end>'
        word_to_idx = dict()
        word_to_idx['<end>'] = 0
        idx = 1

        for w in vocab:
            word_to_idx[w] = idx
            idx_to_word[idx] = w
            idx += 1

        vocabs = {"idx_to_word": idx_to_word, "word_to_idx": word_to_idx}
        with open(self.vocab_path, "wb") as f:
            pickle.dump(vocabs, f)

        return vocabs

    def train_test_split(self, percent=0.1):
        filepath = os.path.join(self.data_dir, 'images.txt')
        df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        labels = np.arange(0, len(filenames))

        X_train, X_test, y_train, y_test = train_test_split(filenames, labels, test_size=percent)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.75*percent)

        file_names = {"train": X_train, "val": X_val, "test": X_test}

        with open(self.train_test_split_path, "wb" ) as tt_file:
            pickle.dump(file_names, tt_file)

        return file_names


class BaseTokenizer(ABC):
    def __init__(self, max_caption_size=10):
        self.max_caption_size =max_caption_size

    def get_padded_tensor(self, caption):
        unpadded = self.tokenize(caption)
        length = len(unpadded)
        if length > self.max_caption_size:
            out = unpadded[:self.max_caption_size]
            length = self.max_caption_size
        else:
            out = [0] * self.max_caption_size
            out[:length] = unpadded

        return torch.LongTensor(out), length

    @abstractmethod
    def tokenize(self, caption):
        pass


class CaptionTokenizer(BaseTokenizer):

    def __init__(self, word_to_idx, idx_to_word=None, max_caption_size=MAX_SEQ_LEN):
        super(CaptionTokenizer, self).__init__(max_caption_size)
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.max_caption_size = max_caption_size

    def tokenize(self, caption):
        cap = caption.replace(u"\ufffd\ufffd", u" ")
        tokenizer = RegexpTokenizer(r'\w+')

        tokens = list(map(lambda t: self.word_to_idx[t],
                          filter(lambda x: len(x) > 0 and x in self.word_to_idx,
                                 tokenizer.tokenize(cap.lower()))))
        return tokens


class BertCaptionTokenizer(BaseTokenizer):
    def __init__(self, word_to_idx, idx_to_word=None, max_caption_size=MAX_SEQ_LEN):
        super().__init__(max_caption_size)
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word

        self.max_caption_size = max_caption_size

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        print("[Bert Tokenizer from pretrained model]")

    def tokenize(self, caption):
        unpadded = self.tokenizer.tokenize(caption)
        tokens = self.tokenizer.convert_tokens_to_ids(unpadded)
        return tokens


def prepare_data(data, device, is_damsm=False):
    if len(data) == 4:
        imgs, captions, caption_lengths, class_ids = data
    elif len(data) == 3:
        imgs, captions, caption_lengths = data
        class_ids = None

    # Sort data by the length in a decreasing order
    caption_lengths, sorted_idx = \
        torch.sort(caption_lengths, 0, True)
    
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_idx]
    
    captions = captions[sorted_idx]

    real_imgs = []

    if not is_damsm:
        for i in range(len(imgs)):
            real_imgs.append(imgs[i].to(device))
    else:
        real_imgs = imgs[-1].to(device)
    
    max_len = MAX_SEQ_LEN
    # captions = captions[:, :max_len]
    captions = captions.squeeze()
    captions = captions.to(device)

    if class_ids is not None:
        class_ids = class_ids[sorted_idx].numpy()

    caption_lengths = caption_lengths.numpy()
    mask = caption_lengths[:,None] > np.arange(max_len)
    input_mask = np.zeros(mask.shape)
    input_mask[mask] = 1
    input_mask = torch.from_numpy(input_mask).squeeze().to(device)

    return (real_imgs, captions, caption_lengths, input_mask, class_ids)


def get_imgs(img_path, imsize, branch_num, transform=None, img=None):
    """
    :param img_path:
    :param imsize: list of the size
    :param transform: transformation without normalize and to_tensor methods
    :return: batch of the images
    """
    normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )  # TODO: appropriate normalization
    if img is None:
        img = Image.open(img_path).convert('RGB') # default for PIL is BGR

    if transform is not None:
        img = transform(img)

    transformed_images = []

    for i in range(branch_num):
        resized_image = transforms.Resize((imsize[i], imsize[i]))(img)
        transformed_images.append(normalize(resized_image))

    return transformed_images


class BirdsDataset(Dataset):
    def __init__(self, mode='test', tokenizer=None, preprocessor=None,
                 base_size=64, branch_num=3, transform=None):
        """
        :param mode: train/test/val
        :param tokenizer: object which can tokenize caption
        :param preprocessor: object with path to train, text, validation and vocabulary
        :param base_size: size of the image in the 1st stage
        :param branch_num: number of the stage (default 3)
        """
        super(BirdsDataset, self).__init__()
        self.mode = mode
        self.transform = transform
        self.max_caption_size = MAX_SEQ_LEN
        if preprocessor is None:
            self.preprocessor = BirdsPreprocessor(data_path='dataset/CUB_200_2011', dataset_name='cub')
        else:
            self.preprocessor = preprocessor
        self.branch_num = branch_num
        self.tokenizer = tokenizer
        self.n_tokens = len(tokenizer.word_to_idx)
        self.imsize = []

        self._load_all_captions()

        for _ in range(self.branch_num):
            self.imsize.append(base_size)
            base_size = base_size * 2

    def __len__(self):
        return len(self.img_file_names)

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
            Tuple: list image (branch_num x [CxWxH]),
                   caption (max_seq len),
                   caption_length (int),
                   class_id (int)
        """
        image_name = os.path.join(self.preprocessor.data_path, self.img_file_names[idx])
        class_id = self.class_ids[idx]
        image = get_imgs(image_name, self.imsize, branch_num=self.branch_num, transform=self.transform)
        # select a random sentence
        cap_idx = np.random.choice(np.arange(len(self.img_captions[idx])))
        caption, caption_length = self.tokenizer.get_padded_tensor(self.img_captions[idx][cap_idx])
        caption_length = np.array(caption_length)

        return image, caption, caption_length, class_id

    def _which_image_data(self):
        img_file_names = []
        if self.mode == "train":
            img_file_names = self.preprocessor.train
        elif self.mode == "val":
            img_file_names = self.preprocessor.val
        else:
            img_file_names = self.preprocessor.test
        return img_file_names

    def _load_all_captions(self):
        self.img_file_names = self._which_image_data()
        self.class_ids = [int(l.split('.')[0]) for l in self.img_file_names]
        self.img_captions = []
        for name in self.img_file_names:
            name_parts = name.split('.')
            txt_name = '.'.join(name_parts[0:-1]) + '.txt'
            txt_path = os.path.join(self.preprocessor.captions_path, txt_name)
            with open(txt_path, encoding='utf-8') as captions_file:
                captions = captions_file.read().splitlines()
                self.img_captions.append(captions)
    
    def tensor_to_caption(self, cap_idx):
        f = lambda x: \
            " ".join([self.preprocessor.idx_to_word[idx.item()]
                      for idx in x if idx.item() != 0])
        output = []

        if cap_idx.dim() == 1:
            output.append(f(cap_idx))
        elif cap_idx.dim() == 2:
            for i in range(cap_idx.size(0)):
                output.append(f(cap_idx[i]))

        return output

    
class CocoPreprocessor(DataPreprocessor):
    def __init__(self, dataset_name, data_path):
        super(CocoPreprocessor, self).__init__(data_path=data_path)

        self.data_dir = data_path
        self.data_path = os.path.join(data_path, "images")
        self.captions_path = os.path.join(data_path, "annotations")
        self.processed_data = "data/"


        self.vocab_path = self.processed_data + dataset_name + "_vocab.pkl"
        self.train_test_split_path = self.processed_data + dataset_name + "_data.pkl"

        if not os.path.exists(self.processed_data):
            os.makedirs(self.processed_data)

        self.vocabs = {"idx_to_word": {}, "word_to_idx": {}}

        if os.path.exists(self.vocab_path):
            with open(self.vocab_path, "rb") as bow_file:
                self.vocabs = pickle.load(bow_file)
        else:
            self.vocabs = self.preprocess()

        self.idx_to_word = self.vocabs["idx_to_word"]
        self.word_to_idx = self.vocabs["word_to_idx"]
        
    def preprocess(self):
        """
        create vocabulary, tokenize captions with len>0
        :return: vocab dict
        """
        train_annotations = os.path.join(self.captions_path, "captions_train2017.json")
        coco_ann = COCO(train_annotations)

        all_captions =[]
        for cap_dict in coco_ann.anns.values():
            cap = cap_dict['caption']
            if len(cap) == 0:
                continue
            cap = cap.replace(u"\ufffd\ufffd", u" ")

            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(cap.lower())

            if len(tokens) == 0:
                print('cap', cap)
                continue

            tokens_new = []
            for t in tokens:
                t = t.encode('ascii', 'ignore').decode('ascii')
                if len(t) > 0:
                    tokens_new.append(t)
            all_captions.extend(tokens_new)

        vocab = np.unique(all_captions)

        idx_to_word = dict()
        idx_to_word[0] = '<end>'
        word_to_idx = dict()
        word_to_idx['<end>'] = 0
        idx = 1

        for w in vocab:
            word_to_idx[w] = idx
            idx_to_word[idx] = w
            idx += 1

        vocabs = {"idx_to_word": idx_to_word, "word_to_idx": word_to_idx}
        with open(self.vocab_path, "wb") as f:
            pickle.dump(vocabs, f)

        return vocabs
    
class CocoDataset(Dataset):
    def __init__(self, mode='val', tokenizer=None, preprocessor=None,
                 base_size=64, branch_num=3, transform=None):
        """
        :param mode: train/test/val
        :param tokenizer: object which can tokenize caption
        :param preprocessor: object with path to train, text, validation and vocabulary
        :param base_size: size of the image in the 1st stage
        :param branch_num: number of the stage (default 3)
        """
        super(CocoDataset, self).__init__()
        self.mode = mode
        self.transform = transform
        self.max_caption_size = MAX_SEQ_LEN
        if preprocessor is None:
            self.preprocessor = CocoPreprocessor(data_path='.', dataset_name='cub')
        else:
            self.preprocessor = preprocessor
        
        data_path = self.preprocessor.data_dir
        self.raw_dataset = CocoCaptions(
            os.path.join(data_path, 'images', '{}2017/'.format(mode)),
            os.path.join(data_path, 'annotations',
                'captions_{}2017.json'.format(mode)
            )
        )
        
        self.branch_num = branch_num
        self.tokenizer = tokenizer
        self.n_tokens = len(tokenizer.word_to_idx)
        self.imsize = []

        for i in range(self.branch_num):
            self.imsize.append(base_size)
            base_size = base_size * 2

    def __len__(self):
        return len(self.raw_dataset)

    def get_test_split_imgs_paths(self):
        return os.path.join(self.preprocessor.data_dir, 'images', 'test2017')

    def __getitem__(self, idx):
        """
        :param idx:
        :return: Tuple: lis image (branch_num x [CxWxH]), caption (max_seq len), caption_length (int)
        """
        raw_image, img_captions = self.raw_dataset[idx]
        image = get_imgs(None, self.imsize, branch_num=self.branch_num, transform=self.transform, img=raw_image)
        # select a random sentence
        cap_idx = np.random.choice(np.arange(len(img_captions)))
        caption, caption_length = self.tokenizer.get_padded_tensor(img_captions[cap_idx])
        caption_length = np.array(caption_length)

        return image, caption, caption_length
    
    def tensor_to_caption(self, cap_idx):
        f = lambda x: \
            " ".join([self.preprocessor.idx_to_word[idx.item()]
                      for idx in x if idx.item() != 0])
        output = []

        if cap_idx.dim() == 1:
            output.append(f(cap_idx))
        elif cap_idx.dim() == 2:
            for i in range(cap_idx.size(0)):
                output.append(f(cap_idx[i]))

        return output
    


if __name__ == '__main__':
    preproc = BirdsPreprocessor(data_path='dataset/CUB_200_2011', dataset_name='cub')
    assert len(preproc.train) == 9813
    assert len(preproc.test) == 1179
    tokenizer = CaptionTokenizer(word_to_idx=preproc.word_to_idx)
    tokenizer = BertCaptionTokenizer(word_to_idx=preproc.word_to_idx)
    test_str = 'it is the caption of the birds'
    test_str2 = 'это подпись к птице'
    print(test_str, tokenizer.tokenize(test_str))
    print(test_str2, tokenizer.tokenize(test_str))

    dataset = BirdsDataset(tokenizer=tokenizer, preprocessor=preproc, branch_num=2)
    image, caption, length, class_id = dataset[0]
    assert image[0].size() == torch.Size([3, 64, 64])
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4)
    next_batch = next(iter(data_loader))
    print(len(next_batch))
    print(len(next_batch[0]))
    assert next_batch[0][0].size() == torch.Size([4, 3, 64, 64])
    
    preproc = CocoPreprocessor(data_path='coco', dataset_name='coco')
    tokenizer = CaptionTokenizer(word_to_idx=preproc.word_to_idx)
    tokenizer = BertCaptionTokenizer(word_to_idx=preproc.word_to_idx)
    test_str = 'it is the caption'
    test_str2 = 'это подпись'
    print(test_str, tokenizer.tokenize(test_str))
    print(test_str2, tokenizer.tokenize(test_str))

    dataset = CocoDataset(tokenizer=tokenizer, preprocessor=preproc, branch_num=2)
    image, caption, length = dataset[0]
    assert image[0].size() == torch.Size([3, 64, 64])
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4)
    next_batch = next(iter(data_loader))
    print(len(next_batch))
    print(len(next_batch[0]))
    assert next_batch[0][0].size() == torch.Size([4, 3, 64, 64])
