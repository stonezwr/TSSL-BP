import math

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.nn.init as init
import functions.tsslbp as tsslbp
import global_v as glv


class LinearLayer(nn.Linear):
    def __init__(self, network_config, config, name, in_shape):
        # extract information for kernel and inChannels
        in_features = config['n_inputs']
        out_features = config['n_outputs']
        self.layer_config = config
        self.network_config = network_config
        self.name = name
        self.type = config['type']
        self.in_shape = in_shape
        self.out_shape = [out_features, 1, 1]
        self.in_spikes = None
        self.out_spikes = None

        if 'weight_scale' in config:
            weight_scale = config['weight_scale']
        else:
            weight_scale = 1

        if type(in_features) == int:
            n_inputs = in_features
        else:
            raise Exception('inFeatures should not be more than 1 dimesnion. It was: {}'.format(in_features.shape))
        if type(out_features) == int:
            n_outputs = out_features
        else:
            raise Exception('outFeatures should not be more than 1 dimesnion. It was: {}'.format(out_features.shape))

        super(LinearLayer, self).__init__(n_inputs, n_outputs, bias=False)

        nn.init.kaiming_normal_(self.weight)
        self.weight = torch.nn.Parameter(weight_scale * self.weight.cuda(), requires_grad=True)
        
        print("linear")
        print(self.name)
        print(self.in_shape)
        print(self.out_shape)
        print(list(self.weight.shape))
        print("-----------------------------------------")

    def forward(self, x):
        """
        """
        x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4])
        x = x.transpose(1, 2)
        y = f.linear(x, self.weight, self.bias)
        y = y.transpose(1, 2)
        y = y.view(y.shape[0], y.shape[1], 1, 1, y.shape[2])
        return y

    def forward_pass(self, x, epoch):
        y = self.forward(x)
        y = tsslbp.TSSLBP.apply(y, self.network_config, self.layer_config)
        return y

    def get_parameters(self):
        return self.weight

    def weight_clipper(self):
        w = self.weight.data
        w = w.clamp(-4, 4)
        self.weight.data = w
