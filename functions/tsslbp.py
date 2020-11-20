import torch
import torch.nn as nn
import torch.nn.functional as f
from time import time 
import global_v as glv


class TSSLBP(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, inputs, network_config, layer_config):
        shape = inputs.shape
        n_steps = glv.n_steps 
        theta_m = 1/network_config['tau_m']
        theta_s = 1/network_config['tau_s']
        threshold = layer_config['threshold']
        warmup_threshold = network_config['warmup']

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

            # record reset effect
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
        ctx.save_for_backward(mem_updates, outputs, mems, delta_refs, torch.tensor([threshold, warmup_threshold]))
        return syns_posts

    @staticmethod
    def backward(ctx, grad_delta):
        # in: grad_output: e(l-1)
        # out: grad: delta(l-1)
        (delta_u, outputs, u, delta_refs, others) = ctx.saved_tensors
        shape = outputs.shape
        n_steps = glv.n_steps
        threshold = others[0].item()
        warmup_threshold = others[1].item()

        if torch.sum(outputs)/(shape[0]*shape[1]*shape[2]*shape[3]*shape[4]) > warmup_threshold:
            partial_u = torch.clamp(-1 / delta_u, -10, 10) * outputs
            partial_u_partial_tp = partial_u.unsqueeze(-1).repeat(1, 1, 1, 1, 1, n_steps)

            # part one, inter-neuron
            partial_a_inter = glv.partial_a.repeat(shape[0], shape[1], shape[2], shape[3], 1, 1)

            # part two, intra-neuron: effect of reset
            partial_a_intra = torch.einsum('...ij, ...jk -> ...ik', delta_refs, partial_a_inter*partial_u_partial_tp)

            # part one, inter-neuron + part two, intra-neuron
            grad_a = torch.einsum('...ij, ...j -> ...i', partial_a_inter + partial_a_intra, grad_delta)

            grad = grad_a * partial_u

        else:
            # warm up
            syn = glv.syn_a.repeat(shape[0], shape[1], shape[2], shape[3], 1, 1)

            grad_a = torch.einsum('...ij, ...j -> ...i', syn, grad_delta)

            a = 0.2
            f = torch.clamp((-1 * u + threshold) / a, -10, 10)
            f = torch.exp(f)
            f = f / ((1 + f) * (1 + f) * a)

            grad = grad_a * f

        return grad, None, None


class TSSLBP_long_time(torch.autograd.Function):  
    @staticmethod
    def forward(ctx, inputs, network_config, layer_config):
        shape = inputs.shape
        n_steps = shape[4] 
        theta_m = 1/network_config['tau_m']
        tau_s = network_config['tau_s']
        theta_s = 1/tau_s
        threshold = layer_config['threshold']
        warmup_threshold = network_config['warmup']

        mem = torch.zeros((shape[0], shape[1], shape[2], shape[3]), dtype=glv.dtype, device=glv.device)
        syn = torch.zeros((shape[0], shape[1], shape[2], shape[3]), dtype=glv.dtype).to(glv.device)
        syns_posts = []
        mems = []
        mem_updates = []
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
        outputs = torch.stack(outputs, dim = 4)
        syns_posts = torch.stack(syns_posts, dim = 4)
        ctx.save_for_backward(mem_updates, outputs, mems, torch.tensor([threshold, tau_s, theta_m, warmup_threshold]))

        return syns_posts

    @staticmethod
    def backward(ctx, grad_delta):
        (delta_u, outputs, u, others) = ctx.saved_tensors
        shape = grad_delta.shape
        n_steps = shape[4]
        threshold = others[0].item()
        tau_s = others[1].item()
        theta_m = others[2].item()
        warmup_threshold = others[3].item()


        grad = torch.zeros_like(grad_delta)

        partial_a_intra = torch.zeros((shape[0], shape[1], shape[2], shape[3], n_steps), dtype=glv.dtype, device=glv.device)
        ref = torch.zeros((shape[0], shape[1], shape[2], shape[3]), dtype=glv.dtype, device=glv.device)

        if torch.sum(outputs)/(shape[0]*shape[1]*shape[2]*shape[3]*shape[4]) > warmup_threshold:
            for t in range(n_steps-1, -1, -1):
                time_end = int(min(t+2*tau_s, n_steps))
                time_len = time_end-t
                out = outputs[..., t]
                partial_u = torch.clamp(-1/delta_u[..., t], -10, 10) * out
                # part one, inter-neuron
                partial_a_inter = glv.partial_a[..., t, t:time_end].repeat(shape[0], shape[1], shape[2], shape[3], 1)
                # inter-neuron + intra-neuron 
                partial_a_all = partial_a_inter + partial_a_intra[..., t:time_end] * (ref * theta_m * delta_u[..., t]  * out).unsqueeze(-1).repeat(1, 1, 1, 1, time_len)

                grad[..., t] = partial_u * torch.sum(partial_a_all*grad_delta[..., t:time_end], dim=4) 
              
                # part two, intra-neuron, current time is t_p
                partial_a_intra[..., t:time_end] = partial_a_intra[..., t:time_end] * (1-out.unsqueeze(-1).repeat(1, 1, 1, 1,time_len)) + partial_a_all * partial_u.unsqueeze(-1).repeat(1, 1, 1, 1,time_len)

                ref = (1 - theta_m) * ref * (1-out) + out
                
        else:
            for t in range(n_steps):
                time_end = int(min(t+2*tau_s, n_steps))
                partial_a_all = glv.syn_a[..., t, t:time_end].repeat(shape[0], shape[1], shape[2], shape[3], 1)

                grad_a = torch.sum(partial_a_all*grad_delta[..., t:time_end], dim=-1)

                a = 0.2
                f = torch.clamp((-1 * u[..., t] + threshold) / a, -8, 8)
                f = torch.exp(f)
                f = f / ((1 + f) * (1 + f) * a)

                grad[..., t] = grad_a * f

        return grad, None, None
    
