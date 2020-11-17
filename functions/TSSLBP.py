import torch
import torch.nn as nn
import torch.nn.functional as f
from time import time 
import global_v as glv


def psp(inputs, network_config):
    shape = inputs.shape
    n_steps = network_config['n_steps']
    tau_s = network_config['tau_s']

    syn = torch.zeros((shape[0], shape[1], shape[2], shape[3]), dtype=glv.dtype, device=glv.device)
    syns = torch.zeros((shape[0], shape[1], shape[2], shape[3], n_steps), dtype=glv.dtype, device=glv.device)

    for t in range(n_steps):
        syn = syn - syn / tau_s + inputs[..., t]
        syns[..., t] = syn / tau_s

    return syns


class TSSLBP(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, inputs, network_config, layer_config):
        shape = inputs.shape
        n_steps = network_config['n_steps']
        theta_m = 1/network_config['tau_m']
        theta_s = 1/network_config['tau_s']
        threshold = layer_config['threshold']

        mem = torch.zeros((shape[0], shape[1], shape[2], shape[3]), dtype=glv.dtype, device=glv.device)
        syn = torch.zeros((shape[0], shape[1], shape[2], shape[3]), dtype=glv.dtype, device=glv.device)
        ref = torch.zeros((shape[0], shape[1], shape[2], shape[3], shape[4]), dtype=glv.dtype, device=glv.device)
        delta_refs = torch.zeros((shape[0], shape[1], shape[2], shape[3], shape[4], shape[4]), dtype=glv.dtype, device=glv.device)
        mems = []
        mem_updates = []
        outputs = []
        syns_posts = []
        outputs = []
        for t in range(n_steps):
            mem_update = (-theta_m) * mem + inputs[..., t]
            delta_ref = (-theta_m) * ref
            mem += mem_update
            ref += delta_ref

            out = mem > threshold
            out = out.type(glv.dtype)
            mems.append(mem)
            if t > 0:
                out_tmp = out.unsqueeze(-1).repeat(1, 1, 1, 1, t-1)
                ref[..., 0:t-1] *= (1-out_tmp)
                delta_ref[..., 0:t-1] *= out_tmp
                delta_refs[..., 0:t-1, t] = delta_ref[..., 0:t-1]
            ref[..., t] = (-1) * mem * out

            mem = mem * (1-out)
            outputs.append(out)
            mem_updates.append(mem_update)

            syn = syn + (out - syn) * theta_s
            syns_posts.append(syn)
        mems = torch.stack(mems, dim = 4)
        mem_updates = torch.stack(mem_updates, dim = 4)
        syns_posts = torch.stack(syns_posts, dim = 4)
        outputs = torch.stack(outputs, dim = 4)
        ctx.save_for_backward(mem_updates, outputs, mems, delta_refs, torch.tensor([threshold]))
        return syns_posts

    @staticmethod
    def backward(ctx, grad_delta):
        # in: grad_output: e(l-1)
        # out: grad: delta(l-1)
        (delta_u, outputs, u, delta_refs, others) = ctx.saved_tensors
        start_time = time()
        shape = outputs.shape
        n_steps = glv.n_steps
        threshold = others[0].item()

        if torch.sum(outputs)/(shape[0] * shape[1] * shape[2] * shape[3] * shape[4]) > 0.1:
            partial_a_inter = glv.partial_a.repeat(shape[0], shape[1], shape[2], shape[3], 1, 1)
            partial_u = torch.clamp(-1 / delta_u, -10, 10) * outputs
            partial_u_partial_tp = partial_u.unsqueeze(-1).repeat(1, 1, 1, 1, 1, n_steps)

            # part two, intra-neuron: effect of reset
            partial_a_intra = torch.einsum('...ij, ...jk -> ...ik', delta_refs, partial_a_inter*partial_u_partial_tp)

            # part one, inter-neuron + part two, intra-neuron
            partial_a_all = partial_a_inter + partial_a_intra

            grad_a = torch.einsum('...ij, ...j -> ...i', partial_a_all, grad_delta)

            grad = grad_a * partial_u

        else:
            # warm up
            syn = glv.syn_a.repeat(shape[0], shape[1], shape[2], shape[3], 1, 1)

            grad_a = torch.einsum('...ij, ...j -> ...i', syn, grad_delta)

            a = 0.2
            f = torch.clamp((-1 * u + threshold) / a, -8, 8)
            f = torch.exp(f)
            f = f / ((1 + f) * (1 + f) * a)

            grad = grad_a * f
        return grad, None, None, None, None, None, None, None, None

