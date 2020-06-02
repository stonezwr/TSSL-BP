import csv
import math
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from os import listdir
from os.path import isfile, join


class NMNIST(Dataset):
    def __init__(self, dataset_path, n_steps, transform=None):
        self.path = dataset_path
        self.samples = []
        self.labels = []
        self.transform = transform
        self.n_steps = n_steps
        for i in tqdm(range(10)):
            sample_dir = dataset_path + '/' + str(i) + '/'
            for f in listdir(sample_dir):
                filename = join(sample_dir, f)
                if isfile(filename):
                    self.samples.append(filename)
                    self.labels.append(i)

    def __getitem__(self, index):
        filename = self.samples[index]
        label = self.labels[index]

        data = np.zeros((2, 34, 34, self.n_steps))

        f = open(filename, 'r')
        lines = f.readlines()
        for line in lines:
            if line is None:
                break
            line = line.split()
            line = [int(l) for l in line]
            pos = line[0] - 1
            if pos >= 1156:
                channel = 1
                pos -= 1156
            else:
                channel = 0
            y = pos % 34
            x = int(math.floor(pos/34))
            for i in range(1, len(line)):
                if line[i] >= self.n_steps:
                    break
                data[channel, x, y, line[i]-1] = 1
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


def get_nmnist(data_path, network_config):
    n_steps = network_config['n_steps']
    batch_size = network_config['batch_size']
    print("loading NMNIST")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    train_path = data_path + '/Train'
    test_path = data_path + '/Test'
    trainset = NMNIST(train_path, n_steps)
    testset = NMNIST(test_path, n_steps)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    return trainloader, testloader
