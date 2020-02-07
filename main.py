import os

import torch
from network_parser import parse
from datasets import loadMNIST, loadCIFAR10, loadFashionMNIST, loadNMNIST
import logging
import cnns
from utils import learningStats
import functions.loss_f as loss_f
import numpy as np
from datetime import datetime
import argparse



def train(network, trainloader, opti, epoch, states, network_config, layers_config, err, is_iow):
    global max_accuracy_train
    global min_loss_train
    logging.info('\nEpoch: %d', epoch)
    train_loss = 0
    correct = 0
    total = 0
    d_type = network_config['dtype']
    dev = network_config['device']
    n_steps = network_config['n_steps']
    time = datetime.now()

    for batch_idx, (in_data, labels) in enumerate(trainloader):
        if "SpikeBasedBPTT" in network_config and network_config["SpikeBasedBPTT"]:
            if is_iow:
                desired_spikes = torch.tensor([1, 2, 3, 3, 3])
            else:
                desired_spikes = torch.tensor([0, 1, 1, 1, 1])
            undesired_spikes = torch.zeros(n_steps).to(device)
            desired_spikes = desired_spikes.view(1, 1, 1, 1, n_steps)
            undesired_spikes = undesired_spikes.view(1, 1, 1, 1, n_steps)
            inputs = in_data.unsqueeze_(-1).repeat(1, 1, 1, 1, n_steps)

            # forward pass
            labels = labels.to(dev)
            inputs = inputs.to(dev)
            inputs.type(d_type)
            opti.zero_grad()
            outputs, tmp = network.forward(inputs, network_config, layers_config, is_iow)

            # set target signal
            targets = undesired_spikes.repeat(outputs.shape[0], outputs.shape[1], outputs.shape[2],
                                              outputs.shape[3], 1)
            for i in range(len(labels)):
                targets[i, labels[i], ...] = desired_spikes

            loss = err.spike_time(outputs, targets, layers_config[list(layers_config.keys())[-1]])

            # backward pass
            loss.backward()
            opti.step()

            # record results
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
        states.training.lossSum += loss.cpu().data.item()  # torch.sum(loss).item()
        states.print(epoch, batch_idx, (datetime.now() - time).total_seconds())

    total_accuracy = correct / total
    total_loss = train_loss / total
    if total_accuracy > max_accuracy_train:
        max_accuracy_train = total_accuracy
    if min_loss_train > total_loss:
        min_loss_train = total_loss

    logging.info("Train Accuracy: %.3f (%.3f). Loss: %.3f (%.3f)\n", 100. * total_accuracy, 100 * max_accuracy_train,
                 total_loss, min_loss_train)


def test(network, testloader, epoch, states, network_config, layers_config, is_iow):
    global best_acc
    global best_epoch
    correct = 0
    total = 0
    dev = network_config['device']
    n_steps = network_config['n_steps']
    time = datetime.now()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(testloader):
            if "SpikeBasedBPTT" in network_config and network_config["SpikeBasedBPTT"]:
                inputs.unsqueeze_(-1)
                inputs = inputs.repeat(1, 1, 1, 1, n_steps)
                # forward pass
                labels = labels.to(dev)
                inputs = inputs.to(dev)
                outputs, tmp = network.forward(inputs, network_config, layers_config, is_iow)

                spike_counts = torch.sum(outputs, dim=4).squeeze_(-1).squeeze_(-1).detach().cpu().numpy()
                predicted = np.argmax(spike_counts, axis=1)
                labels = labels.cpu().numpy()
                total += len(labels)
                correct += (predicted == labels).sum().item()
            else:
                raise Exception('Unrecognized rule name.')

            states.testing.correctSamples += (predicted == labels).sum().item()
            states.testing.numSamples = total
            states.print(epoch, batch_idx, (datetime.now() - time).total_seconds())

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': network.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
        best_epoch = epoch
    logging.info("Test Accuracy: %.3f (%.3f). \n", acc, best_acc)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-config', action='store', dest='config', help='The path of config file')
    parser.add_argument('-checkpoint', action='store', dest='checkpoint', help='The path of checkpoint, if use checkpoint')
    parser.add_argument('-iow', action='store_true', dest='iow', help='enable iow')
    args = parser.parse_args()

    enable_iow = args.iow
    if enable_iow:
        print("IOW is enabled")

    if args.config is None:
        raise Exception('Unrecognized config file.')
    else:
        config_path = args.config


    logging.basicConfig(filename='result.log', level=logging.INFO)

    logging.info("start parsing settings")

    params = parse(config_path)

    logging.info("finish parsing settings")

    dtype = torch.float32

    # Check whether a GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("device is", device)

    params['Network']['dtype'] = dtype
    params['Network']['device'] = device

    logging.info("dataset loaded")
    if params['Network']['dataset'] == "MNIST":
        data_path = os.path.expanduser("mnist")
        train_loader, test_loader = loadMNIST.get_mnist(data_path, params['Network']['batch_size'])
    elif params['Network']['dataset'] == "NMNIST":
        data_path = os.path.expanduser("mnist/NMNIST-Non-Spiking")
        train_loader, test_loader = loadNMNIST.get_nmnist(data_path, params['Network']['batch_size'])
    elif params['Network']['dataset'] == "FashionMNIST":
        data_path = os.path.expanduser("mnist")
        train_loader, test_loader = loadFashionMNIST.get_fashionmnist(data_path,
                                                                           params['Network']['batch_size'])
    elif params['Network']['dataset'] == "CIFAR10":
        data_path = os.path.expanduser("cifar10")
        train_loader, test_loader = loadCIFAR10.get_cifar10(data_path,
                                                            params['Network']['batch_size'])
    else:
        train_loader, test_loader = None, None
        logging.error("dataset do not exist")
        exit(0)
    logging.info("dataset loaded")

    net = cnns.Network(params['Network'], params['Layers'], list(train_loader.dataset[0][0].shape)).to(device)

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])

    error = loss_f.SpikeLoss(params['Network']).to(device)

    parameters = net.get_parameters()
    optimizer = torch.optim.AdamW(parameters, lr=params['Network']['lr'], betas=(0.9, 0.999))

    best_acc = 0
    best_epoch = 0
    max_accuracy_train = 0
    min_loss_train = 1000

    l_states = learningStats()

    for e in range(params['Network']['epochs']):
        l_states.training.reset()
        train(net, train_loader, optimizer, e, l_states, params['Network'], params['Layers'], error, enable_iow)
        l_states.training.update()
        l_states.testing.reset()
        test(net, test_loader, e, l_states, params['Network'], params['Layers'], enable_iow)
        l_states.testing.update()

    logging.info("Best Accuracy: %.3f, at epoch: %d \n", best_acc, best_epoch)
