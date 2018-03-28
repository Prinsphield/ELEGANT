# -*- coding:utf-8 -*-
# Created Time: 2017/12/14 18:38:56
# Author: Taihong Xiao <xiaotaihong@126.com>

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os,time
from PIL import Image
# from scipy import misc
# import skimage.transform
# import warnings
# warnings.filterwarnings("ignore")


class Config:
    @property
    def data_dir(self):
        data_dir = './datasets/celebA'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        return data_dir

    @property
    def exp_dir(self):
        exp_dir = os.path.join('train_log')
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        return exp_dir

    @property
    def model_dir(self):
        model_dir = os.path.join(self.exp_dir, 'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir

    @property
    def log_dir(self):
        log_dir = os.path.join(self.exp_dir, 'log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    @property
    def img_dir(self):
        img_dir = os.path.join(self.exp_dir, 'img')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        return img_dir

    nchw = [16,3,256,256]

    G_lr = 2e-4

    D_lr = 2e-4

    betas = [0,5, 0.999]

    weight_decay = 1e-5

    step_size = 3000

    gamma = 0.97

    shuffle = True

    num_workers = 5

    max_iter = 200000

config = Config()


class SingleCelebADataset(Dataset):
    def __init__(self, im_names, labels, config):
        self.im_names = im_names
        self.labels = labels
        self.config = config

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, idx):
        image = Image.open(self.im_names[idx])
        image = self.transform(image) * 2 - 1
        label = (self.labels[idx] + 1) / 2
        return image, label

    @property
    def transform(self):
        transform = transforms.Compose([
            transforms.Resize(self.config.nchw[-2:]),
            transforms.ToTensor(),
        ])
        return transform

    def gen(self):
        dataloader = DataLoader(self, batch_size=self.config.nchw[0], shuffle=self.config.shuffle, num_workers=self.config.num_workers, drop_last=True)
        while True:
            for data in dataloader:
                yield data

class MultiCelebADataset(object):
    def __init__(self, attributes, config=config):
        self.attributes = attributes
        self.config = config

        with open(os.path.join(self.config.data_dir, 'list_attr_celeba.txt'), 'r') as f:
            lines = f.read().strip().split('\n')
            col_ids = [lines[1].split().index(attribute) + 1 for attribute in self.attributes]
            self.all_labels = np.array([[int(x.split()[col_id]) for col_id in col_ids] for x in lines[2:]], dtype=np.float32)
            self.im_names = np.array([os.path.join(self.config.data_dir, 'align_5p/{:06d}.jpg'.format(idx+1)) for idx in range(len(self.all_labels))])

        self.dict = {i: {True: None, False: None} for i in range(len(self.attributes))}
        for attribute_id in range(len(self.attributes)):
            for is_positive in [True, False]:
                idxs = np.where(self.all_labels[:,attribute_id] == (int(is_positive)*2 - 1))[0]
                im_names = self.im_names[idxs]
                labels = self.all_labels[idxs]
                data_gen = SingleCelebADataset(im_names, labels, self.config).gen()
                self.dict[attribute_id][is_positive] = data_gen

    def gen(self, attribute_id, is_positive):
        data_gen = self.dict[attribute_id][is_positive]
        return data_gen

def test():
    dataset = MultiCelebADataset(['Bangs', 'Smiling'])

    import cProfile
    pr = cProfile.Profile()
    pr.enable()
    for i in range(10):
        if i % 4 == 0:
            images, labels = next(dataset.gen(0, True))
        elif i % 4 == 1:
            images, labels = next(dataset.gen(0,False))
        elif i % 4 == 2:
            images, labels = next(dataset.gen(1,True))
        elif i % 4 == 3:
            images, labels = next(dataset.gen(1,False))
        print(i)
        print(images.shape, labels.shape)
        print(labels.numpy())

    pr.disable()
    from IPython import embed; embed(); exit()
    # pr.print_stats(sort='tottime')


if __name__ == "__main__":
    test()
