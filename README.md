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

# Run the code
## MNIST:
```sh
$ python main.py -config Networks/MNIST_CNN.yaml
$ python main.py -config Networks/MNIST_CNN.yaml -iow  // enable IOW spiking neuron model
```

## FashionMNIST:
```sh
$ python main.py -config Networks/FahsionMNIST_CNN.yaml
$ python main.py -config Networks/FashionMNIST_CNN.yaml -iow  // enable IOW spiking neuron model
$ python main.py -config Networks/FahsionMNIST_400_400.yaml
$ python main.py -config Networks/FashionMNIST_400_400.yaml -iow  // enable IOW spiking neuron model
```

## CIFAR10:
```sh
$ python main.py -config Networks/CIFAR10_CNN.yaml -iow 
$ python main.py -config Networks/CIFAR10_CNN.yaml -checkpoint checkpoint/CIFAR10_ckpt.pth -iow  // load the best result checkpoint
```
