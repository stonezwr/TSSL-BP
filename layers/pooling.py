import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


class PoolLayer(nn.Conv3d):
    def __init__(self, network_config, config, name, in_shape):
        self.name = name
        self.layer_config = config
        self.network_config = network_config
        self.type = config['type']
        kernel_size = config['kernel_size']
        if 'padding' in config:
            padding = config['padding']
        else:
            padding = 0

        if 'stride' in config:
            stride = config['stride']
        else:
            stride = None

        if 'dilation' in config:
            dilation = config['dilation']
        else:
            dilation = 1

        if 'theta' in config:
            theta = config['theta']
        else:
            theta = 1.1
        # kernel
        if type(kernel_size) == int:
            kernel = (kernel_size, kernel_size, 1)
        elif len(kernel_size) == 2:
            kernel = (kernel_size[0], kernel_size[1], 1)
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernel_size.shape))

        # stride
        if stride is None:
            stride = kernel
        elif type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))
        super(PoolLayer, self).__init__(1, 1, kernel, stride, padding, dilation, bias=False)

        self.weight = torch.nn.Parameter(1 * theta * torch.ones(self.weight.shape).cuda(), requires_grad=False)
        self.in_shape = in_shape
        self.out_shape = [in_shape[0], int((in_shape[1] + 2 * padding[0] - kernel[0]) / stride[0] + 1),
                          int((in_shape[2] + 2 * padding[1] - kernel[1]) / stride[1] + 1)]
        print(self.name)
        print(self.in_shape)
        print(self.out_shape)
        print(list(self.weight.shape))
        print("-----------------------------------------")

    def forward(self, x):
        result = f.conv3d(x.reshape((x.shape[0], 1, x.shape[1] * x.shape[2], x.shape[3], x.shape[4])),
                          self.weight, self.bias,
                          self.stride, self.padding, self.dilation)
        return result.reshape((result.shape[0], x.shape[1], -1, result.shape[3], result.shape[4]))

    def get_parameters(self):
        return self.weight

    def forward_pass(self, x, epoch):
        y1 = self.forward(x)
        return y1

    def weight_clipper(self):
        return
