# TSSL-BP
This repo is the Pytorch implementation of SNNs trained by the Temporal Spike Sequence Learning via Backpropagation (TSSL-BP).

# Dependencies and Libraries
* python 3.7
* pytorch
* torchvision

# Installation
```sh
$ pip install -r requirements
```

# Preprocessing
## N-MNIST
To reduce the time resolution of the original N-MNIST samples, move and run the NMNIST_Converter.m in the same directory of N-MNIST dataset. 
Parameters: use_two_channels = 1, time_window = 3000.

# Run the code
## Before running
Modify the data path and network settings in the config files of the Networks folder. 
Select the index of GPU (0 by default)

## MNIST:
```sh
$ python main.py -config Networks/MNIST_CNN.yaml
$ python main.py -config Networks/MNIST_CNN.yaml -checkpoint checkpoint/ckpt.pth // load the checkpoint
```

## N-MNIST:
```sh
$ python main.py -config Networks/NMNIST_CNN.yaml
$ python main.py -config Networks/NMNIST_CNN.yaml -checkpoint checkpoint/ckpt.pth // load the checkpoint
```

## FashionMNIST:
```sh
$ python main.py -config Networks/FahsionMNIST_400_400.yaml
$ python main.py -config Networks/FahsionMNIST_400_400.yaml -checkpoint checkpoint/ckpt.pth // load the checkpoint
$ python main.py -config Networks/FahsionMNIST_CNN.yaml
$ python main.py -config Networks/FahsionMNIST_CNN.yaml -checkpoint checkpoint/ckpt.pth // load with checkpoint
```

## CIFAR10:
```sh
$ python main.py -config Networks/CIFAR10_CNN.yaml
$ python main.py -config Networks/CIFAR10_CNN.yaml -checkpoint checkpoint/ckpt.pth   // load the checkpoint
```
