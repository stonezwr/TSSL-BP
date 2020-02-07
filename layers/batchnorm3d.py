import torch.nn as nn


class Batch3DLayer(nn.BatchNorm3d):
    def __init__(self, config, name):
        self.name = name
        self.type = "batch_norm3d"
        if "n_outputs" in config:
            num_features = config["n_outputs"]
        else:
            num_features = config["out_channels"]

        super(Batch3DLayer, self).__init__(num_features)
