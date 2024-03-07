import os
import sys
import pickle

import matplotlib.pyplot as plt
import numpy
import torch
from torch.utils.data import Dataset


class Train_dataset(Dataset):
    def __init__(self, path, transform=None):
        with open(os.path.join(path, 'train'), 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        self.transform = transform
        self.img = data[b'data']
        self.label = data[b'fine_labels']

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        label = self.label[index]
        # ~1024 : R채널, 1024~2048 : G채널, 2048~3072 : B채널
        r = self.img[index, :1024].reshape(32, 32)
        g = self.img[index, 1024:2048].reshape(32, 32)
        b = self.img[index, 2048:].reshape(32, 32)
        # dstack 함수는 열 방향으로 값을 붙여주는
        ''' a = np.array([1, 2, 3])
            b = np.array([4, 5, 6])

            np.dstack((a, b)) # shape=(1, 3, 2)
            # array([[[1, 4],
            #         [2, 5],
            #         [3, 6]]])'''
        img = numpy.dstack((r, g, b))

        if self.transform:
            img = self.transform(img)

        return img, label


class Test_dataset(Dataset):

    def __init__(self, path, transform=None):
        with open(os.path.join(path, 'test'), 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        self.img = data[b'data']
        self.label = data[b'fine_labels']
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        label = self.label[index]
        r = self.img[index, :1024].reshape(32, 32)
        g = self.img[index, 1024:2048].reshape(32, 32)
        b = self.img[index, 2048:].reshape(32, 32)
        img = numpy.dstack((r, g, b))

        if self.transform:
            img = self.transform(img)

        return img, label