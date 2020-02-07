import csv
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from os import listdir
from os.path import isfile, join


class NMNIST(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.path = dataset_path
        self.samples = []
        self.labels = []
        self.transform = transform
        for i in tqdm(range(10)):
            sample_dir = dataset_path + '/' + str(i) + '/'
            for f in listdir(sample_dir):
                filename = join(sample_dir, f)
                if isfile(filename):
                    self.samples.append(filename)
                    self.labels.append(i)

    def __getitem__(self, index):
        data_path = self.samples[index]
        label = self.labels[index]
        tmp = np.genfromtxt(data_path, delimiter=',')

        data = np.zeros((2, 34, 34, 5))
        for c in range(2):
            for row in range(34):
                for col in range(34):
                    data[c, row, col, :] = tmp[c*1156 + row * 34 + col, :]
        if self.transform:
            data = self.transform(data)
            data = data.type(torch.float32)
        else:
            data = torch.FloatTensor(data)

        # Input spikes are reshaped to ignore the spatial dimension and the neurons are placed in channel dimension.
        # The spatial dimension can be maintained and used as it is.
        # It requires different definition of the dense layer.
        return data, label

    def __len__(self):
        return len(self.samples)


def get_nmnist(data_path, batch_size):
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    train_path = data_path + '/Train'
    test_path = data_path + '/Test'
    trainset = NMNIST(train_path) 
    testset = NMNIST(test_path)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    return trainloader, testloader
