# Temporal Spike Sequence Learning via Backpropagation for Deep Spiking Neural Networks (TSSL-BP)

This repository is the official implementation of [Temporal Spike Sequence Learning via Backpropagation for Deep Spiking Neural Networks](https://arxiv.org/abs/2002.10085). 

## Requirements
### Dependencies and Libraries
* python 3.7
* pytorch
* torchvision

### Installation
To install requirements:

```setup
pip install -r requirements.txt
```

### Datasets
NMNIST: [dataset](https://www.garrickorchard.com/datasets/n-mnist), [preprocessing](https://github.com/stonezwr/TSSL-BP/tree/master/preprocessing/NMNIST)

## Training
### Before running
Modify the data path and network settings in the [config files](https://github.com/stonezwr/TSSL-BP/tree/master/Networks). 

Select the index of GPU in the [main.py](https://github.com/stonezwr/TSSL-BP/blob/master/main.py#L198) (0 by default)

### Run the code
```sh
$ python main.py -config Networks/config_file.yaml
$ python main.py -config Networks/config_file.yaml -checkpoint checkpoint/ckpt.pth // load the checkpoint
```

## Results
Our proposed method achieves the following performance on :

### MNIST

| Network Size         | Time Steps | Epochs | Mean | Stddev | Best |
| ------------------ |---------------- | -------------- | ------------- | ------------- | ------------- | 
| 15C5-P2-40C5-P2-300   |     5         |     100      |  99.50% | 0.02% |  99.53% |

### N-MNIST
| Network Size         | Time Steps | Epochs | Mean | Stddev | Best |
| ------------------ |---------------- | -------------- | ------------- | ------------- | ------------- | 
| 12C5-P2-64C5-P2   |     100         |     100      |  99.35% | 0.03% |  99.40% |
| 12C5-P2-64C5-P2   |     30         |     100      |  99.23% | 0.05% |  99.28% |

### Fashion MNIST
| Network Size         | Time Steps | Epochs | Mean | Stddev | Best |
| ------------------ |---------------- | -------------- | ------------- | ------------- | ------------- | 
| 400 − 400  |     5        |     100      |  89.75% | 0.03% |  89.92% |
| 32C5-P2-64C5-P2-1024   |     5         |     100      |  92.69% | 0.09% |  92.83% |

### CIFAR 10
| Network Size         | Time Steps | Epochs | Mean | Stddev | Best |
| ------------------ |---------------- | -------------- | ------------- | ------------- | ------------- | 
| 96C3-256C3-P2-384C3-P2-384C3-256C3-1024-1024  |     5        |     150      |  88.98% | 0.27% |  89.37% |
| 128C3-256C3-P2-512C3-P2-1024C3-512C3-1024-512   |     5         |     150      |  N/A | N/A |  91.41% |
