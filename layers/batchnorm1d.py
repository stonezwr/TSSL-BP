import torch.nn as nn


class Batch1DLayer(nn.BatchNorm1d):
    def __init__(self, config, name):
        self.name = name
        self.type = "batch_norm1d"
        num_features = config["n_outputs"]

        super(Batch1DLayer, self).__init__(num_features)

