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


class PSP_spike_large_batch(torch.autograd.Function): 
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
                out_tmp = out.unsqueeze(-1).repeat(1, 1, 1, 1, t)
                ref[..., 0:t] *= (1-out_tmp)
                delta_ref[..., 0:t] *= out_tmp
            ref[..., t] = (-1) * mem * out
            delta_refs[..., 0:t, t] = delta_ref[..., 0:t]

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
            mini_batch = shape[0]
            partial_a_inter = glv.partial_a.repeat(mini_batch, shape[1], shape[2], shape[3], 1, 1)
            grad_a = torch.empty_like(delta_u)

            for i in range(int(shape[0]/mini_batch)):
                # part two, intra-neuron: effect of reset
                delta_refs_batch = delta_refs[i*mini_batch:(i+1)*mini_batch, ...]
                partial_a_intra = torch.einsum('...ij, ...jk -> ...ik', partial_a_inter, delta_refs_batch)

                # part one, inter-neuron + part two, intra-neuron
                partial_a_all = partial_a_inter + partial_a_intra

                grad_a[i*mini_batch:(i+1)*mini_batch, ...] = torch.einsum('...ij, ...j -> ...i', partial_a_all, grad_delta[i*mini_batch:(i+1)*mini_batch, ...])

            partial_u = torch.clamp(-1 / delta_u, -10, 10) * outputs
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


class PSP_spike_long_time(torch.autograd.Function):  
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
                out_tmp = out.unsqueeze(-1).repeat(1, 1, 1, 1, t)
                ref[..., 0:t] *= (1-out_tmp)
                delta_ref[..., 0:t] *= out_tmp
            ref[..., t] = (-1) * mem * out
            delta_refs[..., 0:t, t] = delta_ref[..., 0:t]

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
        (delta_u, outputs, u, delta_refs, others) = ctx.saved_tensors
        start_time = time()
        shape = outputs.shape
        n_steps = glv.n_steps
        threshold = others[0].item()

        grad_a = torch.empty_like(delta_u)
        
        if torch.sum(outputs)/(shape[0] * shape[1] * shape[2] * shape[3] * shape[4]) > 0.1:
            for t in range(n_steps):
                # part one, inter-neuron
                partial_a_inter = glv.partial_a[..., t, :].repeat(shape[0], shape[1], shape[2], shape[3], 1)

                # part two, intra-neuron: effect of reset
                partial_a_intra = torch.einsum('...j, ...jk -> ...k', partial_a_inter, delta_refs)

                # part one, inter-neuron + part two, intra-neuron
                partial_a_all = partial_a_inter + partial_a_intra

                grad_a[..., t] = torch.sum(partial_a_all[..., t:n_steps]*grad_delta[..., t:n_steps], dim=4) 

            partial_u = torch.clamp(-1 / delta_u, -10, 10) * outputs
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
    

class PSP_spike_fast(torch.autograd.Function):  

    @staticmethod
    def forward(ctx, inputs, network_config, layer_config):
        shape = inputs.shape
        n_steps = network_config['n_steps']
        theta_m = 1/network_config['tau_m']
        theta_s = 1/network_config['tau_s']
        threshold = layer_config['threshold']

        mem = torch.zeros((shape[0], shape[1], shape[2], shape[3]), dtype=glv.dtype, device=glv.device)
        syn = torch.zeros((shape[0], shape[1], shape[2], shape[3]), dtype=glv.dtype, device=glv.device)
        mems = []
        mem_updates = []
        outputs = []
        syns_posts = []
        outputs = []
        for t in range(n_steps):
            mem_update = (-theta_m) * mem + inputs[..., t]
            mem += mem_update

            out = mem > threshold
            out = out.type(glv.dtype)
            mems.append(mem)
            mem = mem * (1-out)
            outputs.append(out)
            mem_updates.append(mem_update)

            syn = syn + (out - syn) * theta_s
            syns_posts.append(syn)
        mems = torch.stack(mems, dim = 4)
        mem_updates = torch.stack(mem_updates, dim = 4)
        syns_posts = torch.stack(syns_posts, dim = 4)
        outputs = torch.stack(outputs, dim = 4)
        ctx.save_for_backward(mem_updates, outputs, mems, torch.tensor([threshold, network_config['tau_s']]))
        return syns_posts

    @staticmethod
    def backward(ctx, grad_delta):
        # in: grad_output: e(l-1)
        # out: grad: delta(l-1)
        (delta_u, outputs, u, others) = ctx.saved_tensors
        start_time = time()
        shape = outputs.shape
        n_steps = glv.n_steps
        threshold = others[0].item()
        tau_s = others[1].item()

        if torch.sum(outputs)/(shape[0] * shape[1] * shape[2] * shape[3] * shape[4]) > 0.05:
            partial_a_inter = glv.partial_a.repeat(shape[0], shape[1], shape[2], shape[3], 1, 1)

            grad_a = torch.einsum('...ij, ...j -> ...i', partial_a_inter, grad_delta)

            partial_u = torch.clamp(-1 / delta_u, -10, 10) * outputs
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

