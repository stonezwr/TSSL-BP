import torch
import torch.nn as nn
import layers.conv as conv
import layers.pooling as pooling
import layers.dropout as dropout
import layers.batchnorm3d as batch_norm3d
import layers.batchnorm1d as batch_norm1d
import layers.linear as linear
import functions.TSSLBP as f


class Network(nn.Module):
    def __init__(self, network_config, layers_config, input_shape):
        super(Network, self).__init__()
        self.layers = []
        self.bn = []
        parameters = []
        print("Network Structure:")
        for key in layers_config:
            c = layers_config[key]
            if c['type'] == 'convspiking':
                self.layers.append(conv.ConvLayer(c, key, input_shape))
                self.layers[-1].to(network_config['device'])
                input_shape = self.layers[-1].out_shape
                parameters.append(self.layers[-1].get_parameters())
                if c['batch_norm3d']:
                    self.bn.append(batch_norm3d.Batch3DLayer(c, key, input_shape, network_config['n_steps']))
                    self.bn[-1].to(network_config['device'])
            elif c['type'] == 'pooling':
                self.layers.append(pooling.PoolLayer(c, key, input_shape))
                self.layers[-1].to(network_config["device"])
                input_shape = self.layers[-1].out_shape
            elif c['type'] == 'spiking':
                self.layers.append(dense.DenseLayer(c, key))
                self.layers[-1].to(network_config['device'])
                parameters.append(self.layers[-1].get_parameters())
                if c['batch_norm3d']:
                    self.bn.append(batch_norm3d.Batch3DLayer(c, key, input_shape, network_config['n_steps']))
                    self.bn[-1].to(network_config['device'])
            elif c['type'] == 'linear':
                self.layers.append(linear.LinearLayer(c, key, network_config['dtype'], network_config['device'],
                                                      input_shape))
                self.layers[-1].to(network_config['device'])
                input_shape = self.layers[-1].out_shape
                parameters.append(self.layers[-1].get_parameters())
                if c['batch_norm1d']:
                    self.bn.append(batch_norm1d.Batch1DLayer(c, key, input_shape, network_config['n_steps']))
                    self.bn[-1].to(network_config['device'])
            elif c['type'] == 'dropout':
                # continue
                self.layers.append(dropout.DropoutLayer(c, key))
            else:
                raise Exception('Undefined layer type. It is: {}'.format(c['type']))
        self.my_parameters = nn.ParameterList(parameters)
        print("-----------------------------------------")

    def forward(self, spike_input, network_config, layer_config, is_iow):
        spikes = spike_input
        fire_count = None
        j = 0
        for i in range(len(self.layers)):
            if self.layers[i].type == "dropout":
                spikes = self.layers[i](spikes)
            elif "SpikeBasedBPTT" in network_config and network_config["SpikeBasedBPTT"]:
                spikes = self.layers[i](spikes)
                if self.layers[i].type != "pooling" and self.layers[i].batch_norm:
                    if self.bn[j].type == "batch_norm3d":
                        spikes = self.bn[j](spikes)
                    elif self.bn[j].type == "batch_norm1d":
                        spikes = self.bn[j](spikes)
                    j += 1

                spikes = f.PSP_spike.apply(spikes, network_config, layer_config[self.layers[i].name], is_iow)
            else:
                raise Exception('Unrecognized rule type. It is: {}'.format(network_config['rule']))
        return spikes, fire_count

    def get_parameters(self):
        return self.my_parameters

