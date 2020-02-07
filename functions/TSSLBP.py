import torch


def psp(inputs, network_config, layer_config):
    shape = inputs.shape
    device = network_config['device']
    dtype = network_config['dtype']
    n_steps = network_config['n_steps']
    tau_s = layer_config['tau_s']

    syn = torch.zeros((shape[0], shape[1], shape[2], shape[3]), dtype=dtype).to(device)
    syns = torch.zeros((shape[0], shape[1], shape[2], shape[3], n_steps), dtype=dtype).to(device)

    for t in range(n_steps):
        syn = syn - syn / tau_s + inputs[..., t]
        syns[..., t] = syn / tau_s

    return syns


class PSP_spike(torch.autograd.Function):  # a and u is the incremnet of each time steps
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    """

    @staticmethod
    def forward(ctx, inputs, network_config, layer_config, is_iow):
        shape = inputs.shape
        device = network_config['device']
        dtype = network_config['dtype']
        n_steps = network_config['n_steps']
        tau_m = layer_config['tau_m']
        tau_s = layer_config['tau_s']
        threshold = layer_config['threshold']

        mem = torch.zeros((shape[0], shape[1], shape[2], shape[3]), dtype=dtype).to(device)
        outputs = torch.zeros((shape[0], shape[1], shape[2], shape[3], n_steps), dtype=dtype).to(device)
        mem_updates = torch.zeros((shape[0], shape[1], shape[2], shape[3], n_steps), dtype=dtype).to(device)
        mems = torch.zeros((shape[0], shape[1], shape[2], shape[3], n_steps), dtype=dtype).to(device)
        syn_updates = torch.zeros((shape[0], shape[1], shape[2], shape[3], n_steps, n_steps), dtype=dtype).to(device)
        partial_a = torch.zeros((shape[0], shape[1], shape[2], shape[3], n_steps, n_steps), dtype=dtype).to(device)
        syn = torch.zeros((shape[0], shape[1], shape[2], shape[3]), dtype=dtype).to(device)
        syns_posts = torch.zeros((shape[0], shape[1], shape[2], shape[3], n_steps), dtype=dtype).to(device)

        for t in range(n_steps):
            out = torch.zeros((shape[0], shape[1], shape[2], shape[3]), dtype=dtype).to(device)
            mem_update = (-1 / tau_m) * mem + inputs[..., t]
            mem += mem_update

            if is_iow:
                out[mem > 0] = torch.floor(mem[mem > 0] / threshold)
                out[out > 3] = 3
            else:
                out[mem > threshold] = 1
            outputs[..., t] = out
            mems[..., t] = mem
            mem[out > 0] = 0
            mem_update[out == 0] = 0
            mem_updates[..., t] = mem_update
            if t > 0:
                syn_updates[..., t] = syn_updates[..., t - 1] - syn_updates[..., t - 1] / tau_s
                partial_a[..., t] = partial_a[..., t - 1] - partial_a[..., t - 1] / tau_s
            syn_updates[..., t, t] += out / tau_s
            partial_a[..., t, t] += 1 / tau_s
            syn = syn - syn / tau_s + out
            syns_posts[..., t] = syn / tau_s
        ctx.save_for_backward(mem_updates, syn_updates, outputs, mems, partial_a, torch.tensor([threshold, tau_m]))

        return syns_posts

    @staticmethod
    def backward(ctx, grad_output):
        # in: grad_output: e(l-1)
        # out: grad: delta(l-1)
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        """
        (delta_u, partial_a, outputs, u, partial_a_2, others) = ctx.saved_tensors
        dtype = delta_u.dtype
        shape = delta_u.shape
        n_steps = shape[4]
        device = delta_u.device
        threshold = others[0].item()
        tau_m = others[1].item()

        e = grad_output.clone()

        f = threshold - u
        f[f > 4] = 4
        f = torch.exp(f)
        f = f / ((1 + f) * (1 + f))

        f = f.unsqueeze_(-1).repeat(1, 1, 1, 1, 1, n_steps)
        partial_a_partial_u_f = partial_a_2 * f

        partial_u = delta_u

        partial_u[partial_u != 0] = 1 / partial_u[partial_u != 0]
        partial_u[partial_u > 10] = 10
        partial_u[partial_u < -10] = -10
        partial_u.unsqueeze_(-1)
        partial_u = partial_u.repeat(1, 1, 1, 1, 1, n_steps)

        partial_a_partial_u = partial_a * partial_u

        # part three, if there's no spike
        base = partial_a_partial_u[..., n_steps - 1, :]
        for t in range(n_steps - 1):
            base = base * (1 - 1 / tau_m)
            index = n_steps - t - 2
            current = partial_a_partial_u[..., index, :]
            current[current == 0] = base[current == 0]
            base[current != 0] = current[current != 0]
            partial_a_partial_u[..., index, :] = current

        partial_a_partial_u[partial_a_partial_u == 0] = partial_a_partial_u_f[partial_a_partial_u == 0]

        grad = torch.einsum('...ij, ...j -> ...i', partial_a_partial_u, e)
        
        return grad, None, None, None, None

