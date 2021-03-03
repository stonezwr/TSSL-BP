import os

import torch
import torch.backends.cudnn as cudnn
from network_parser import parse
from datasets import loadMNIST, loadCIFAR10, loadFashionMNIST, loadNMNIST_Spiking 
import logging
import cnns
from utils import learningStats
from utils import aboutCudaDevices
from utils import EarlyStopping
import functions.loss_f as loss_f
import numpy as np
from datetime import datetime
import pycuda.driver as cuda
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils import clip_grad_value_
import global_v as glv

from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import argparse


max_accuracy = 0
min_loss = 1000


def train(network, trainloader, opti, epoch, states, network_config, layers_config, err):
    network.train()
    global max_accuracy
    global min_loss
    logging.info('\nEpoch: %d', epoch)
    train_loss = 0
    correct = 0
    total = 0
    n_steps = network_config['n_steps']
    n_class = network_config['n_class']
    batch_size = network_config['batch_size']
    time = datetime.now()

    if network_config['loss'] == "kernel":
        # set target signal
        if n_steps >= 10:
            desired_spikes = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]).repeat(int(n_steps/10))
        else:
            desired_spikes = torch.tensor([0, 1, 1, 1, 1]).repeat(int(n_steps/5))
        desired_spikes = desired_spikes.view(1, 1, 1, 1, n_steps).cuda()
        desired_spikes = loss_f.psp(desired_spikes, network_config).view(1, 1, 1, n_steps)
    des_str = "Training @ epoch " + str(epoch)
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        start_time = datetime.now()
        targets = torch.zeros(labels.shape[0], n_class, 1, 1, n_steps).cuda() 
        if network_config["rule"] == "TSSLBP":
            if len(inputs.shape) < 5:
                inputs = inputs.unsqueeze_(-1).repeat(1, 1, 1, 1, n_steps)
            # forward pass
            labels = labels.cuda()
            inputs = inputs.cuda()
            inputs.type(torch.float32)
            outputs = network.forward(inputs, epoch, True)

            if network_config['loss'] == "count":
                # set target signal
                desired_count = network_config['desired_count']
                undesired_count = network_config['undesired_count']

                targets = torch.ones(outputs.shape[0], outputs.shape[1], 1, 1).cuda() * undesired_count
                for i in range(len(labels)):
                    targets[i, labels[i], ...] = desired_count
                loss = err.spike_count(outputs, targets, network_config, layers_config[list(layers_config.keys())[-1]])
            elif network_config['loss'] == "kernel":
                targets.zero_()
                for i in range(len(labels)):
                    targets[i, labels[i], ...] = desired_spikes
                loss = err.spike_kernel(outputs, targets, network_config)
            elif network_config['loss'] == "softmax":
                # set target signal
                loss = err.spike_soft_max(outputs, labels)
            else:
                raise Exception('Unrecognized loss function.')

            # backward pass
            opti.zero_grad()

            loss.backward()
            clip_grad_norm_(network.get_parameters(), 1)
            opti.step()
            network.weight_clipper()

            spike_counts = torch.sum(outputs, dim=4).squeeze_(-1).squeeze_(-1).detach().cpu().numpy()
            predicted = np.argmax(spike_counts, axis=1)
            train_loss += torch.sum(loss).item()
            labels = labels.cpu().numpy()
            total += len(labels)
            correct += (predicted == labels).sum().item()
        else:
            raise Exception('Unrecognized rule name.')

        states.training.correctSamples = correct
        states.training.numSamples = total
        states.training.lossSum += loss.cpu().data.item() 
        states.print(epoch, batch_idx, (datetime.now() - time).total_seconds())

    total_accuracy = correct / total
    total_loss = train_loss / total
    if total_accuracy > max_accuracy:
        max_accuracy = total_accuracy
    if min_loss > total_loss:
        min_loss = total_loss

    logging.info("Train Accuracy: %.3f (%.3f). Loss: %.3f (%.3f)\n", 100. * total_accuracy, 100 * max_accuracy, total_loss, min_loss)


def test(network, testloader, epoch, states, network_config, layers_config, early_stopping):
    network.eval()
    global best_acc
    global best_epoch
    correct = 0
    total = 0
    n_steps = network_config['n_steps']
    n_class = network_config['n_class']
    time = datetime.now()
    y_pred = []
    y_true = []
    des_str = "Testing @ epoch " + str(epoch)
    for batch_idx, (inputs, labels) in enumerate(testloader):
        if network_config["rule"] == "TSSLBP":
            if len(inputs.shape) < 5:
                inputs = inputs.unsqueeze_(-1).repeat(1, 1, 1, 1, n_steps)
            # forward pass
            labels = labels.cuda()
            inputs = inputs.cuda()
            outputs = network.forward(inputs, epoch, False)

            spike_counts = torch.sum(outputs, dim=4).squeeze_(-1).squeeze_(-1).detach().cpu().numpy()
            predicted = np.argmax(spike_counts, axis=1)
            labels = labels.cpu().numpy()
            y_pred.append(predicted)
            y_true.append(labels)
            total += len(labels)
            correct += (predicted == labels).sum().item()
        else:
            raise Exception('Unrecognized rule name.')

        states.testing.correctSamples += (predicted == labels).sum().item()
        states.testing.numSamples = total
        states.print(epoch, batch_idx, (datetime.now() - time).total_seconds())

    test_accuracy = correct / total
    if test_accuracy > best_acc:
        best_acc = test_accuracy
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        cf = confusion_matrix(y_true, y_pred, labels=np.arange(n_class))
        df_cm = pd.DataFrame(cf, index = [str(ind*25) for ind in range(n_class)], columns=[str(ind*25) for ind in range(n_class)])
        plt.figure()
        sn.heatmap(df_cm, annot=True)
        plt.savefig("confusion_matrix.png")
        plt.close()

    logging.info("Train Accuracy: %.3f (%.3f).\n", 100. * test_accuracy, 100 * best_acc)
    # Save checkpoint.
    acc = 100. * correct / total
    early_stopping(acc, network, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', action='store', dest='config', help='The path of config file')
    parser.add_argument('-checkpoint', action='store', dest='checkpoint', help='The path of checkpoint, if use checkpoint')
    parser.add_argument('-gpu', type=int, default=0, help='GPU device to use (default: 0)')
    parser.add_argument('-seed', type=int, default=3, help='random seed (default: 3)')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(0)

    if args.config is None:
        raise Exception('Unrecognized config file.')
    else:
        config_path = args.config

    logging.basicConfig(filename='result.log', level=logging.INFO)
    
    logging.info("start parsing settings")
    
    params = parse(config_path)
    
    logging.info("finish parsing settings")
    
    # check GPU
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    
    # set GPU
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    glv.init(params['Network']['n_steps'], params['Network']['tau_s'] )
    
    logging.info("dataset loaded")
    if params['Network']['dataset'] == "MNIST":
        data_path = os.path.expanduser(params['Network']['data_path'])
        train_loader, test_loader = loadMNIST.get_mnist(data_path, params['Network'])
    elif params['Network']['dataset'] == "NMNIST_Spiking":
        data_path = os.path.expanduser(params['Network']['data_path'])
        train_loader, test_loader = loadNMNIST_Spiking.get_nmnist(data_path, params['Network'])
    elif params['Network']['dataset'] == "FashionMNIST":
        data_path = os.path.expanduser(params['Network']['data_path'])
        train_loader, test_loader = loadFashionMNIST.get_fashionmnist(data_path, params['Network'])
    elif params['Network']['dataset'] == "CIFAR10":
        data_path = os.path.expanduser(params['Network']['data_path'])
        train_loader, test_loader = loadCIFAR10.get_cifar10(data_path, params['Network'])
    else:
        raise Exception('Unrecognized dataset name.')
    logging.info("dataset loaded")
    
    net = cnns.Network(params['Network'], params['Layers'], list(train_loader.dataset[0][0].shape)).cuda()
    
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
    
    error = loss_f.SpikeLoss(params['Network']).cuda()
    
    optimizer = torch.optim.AdamW(net.get_parameters(), lr=params['Network']['lr'], betas=(0.9, 0.999))
    
    best_acc = 0
    best_epoch = 0
    
    l_states = learningStats()
    early_stopping = EarlyStopping()
    
    for e in range(params['Network']['epochs']):
        l_states.training.reset()
        train(net, train_loader, optimizer, e, l_states, params['Network'], params['Layers'], error)
        l_states.training.update()
        l_states.testing.reset()
        test(net, test_loader, e, l_states, params['Network'], params['Layers'], early_stopping)
        l_states.testing.update()
        # if early_stopping.early_stop:
        #     break
    
    logging.info("Best Accuracy: %.3f, at epoch: %d \n", best_acc, best_epoch)
