import os
import torchvision.datasets
import torchvision.transforms as transforms
import torch


def get_mnist(data_path, batch_size):
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])

    trainset = torchvision.datasets.MNIST(data_path, train=True, transform=transform, download=True)
    testset = torchvision.datasets.MNIST(data_path, train=False, transform=transform, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    return trainloader, testloader

