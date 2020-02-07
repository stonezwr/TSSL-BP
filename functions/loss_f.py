import torch
import functions.TSSLBP as f


class SpikeLoss(torch.nn.Module):
    """
    This class defines different spike based loss modules that can be used to optimize the SNN.
    """
    def __init__(self, network_config):
        super(SpikeLoss, self).__init__()
        self.network_config = network_config

    def spike_time(self, spike_out, target, last_layer_config):
        target_a = f.psp(target, self.network_config, last_layer_config)
        return 1 / 2 * torch.sum((spike_out - target_a) ** 2)

