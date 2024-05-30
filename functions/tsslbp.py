import torch
import torch.nn as nn
import torch.nn.functional as f
from time import time 
import global_v as glv


class TSSLBP(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, inputs, network_config, layer_config):
        shape = inputs.shape
        n_steps = shape[4] 
        theta_m = 1/network_config['tau_m']
        tau_s = network_config['tau_s']
        theta_s = 1/tau_s
        threshold = layer_config['threshold']

        mem = torch.zeros(shape[0], shape[1], shape[2], shape[3]).cuda()
        syn = torch.zeros(shape[0], shape[1], shape[2], shape[3]).cuda() #membrane voltages

        syns_posts = []

        mems = []
        mem_updates = []
        outputs = []

        for t in range(n_steps):
            mem_update = (-theta_m) * mem + inputs[..., t]
            mem += mem_update

            out = mem > threshold
            out = out.type(torch.float32)

            mems.append(mem)

            mem = mem * (1-out) #if spikes, membrane voltage back to 0
            outputs.append(out)
            mem_updates.append(mem_update)

            #original: 
            #syn = syn + (out - syn) * theta_s
            
            #new: syns_posts of all time steps before t
            #t = current time step, n = total time steps, theta = decay
            
            count = n_steps - t
            ind = 0
            sum = 0
            while (count > 0):
                sum += glv.syn_a[..., ind] #theta_s is decay, we keep multiplying the decay for every time step
                ind += 1
                count -= 1
            alpha = 1 / sum
            print(t, alpha)
            syn = syn - (syn * theta_s) + out * alpha #multiply 'out' by our constant
            print(t, syn)
            syns_posts.append(syn)


        mems = torch.stack(mems, dim = 4)
        mem_updates = torch.stack(mem_updates, dim = 4)
        outputs = torch.stack(outputs, dim = 4)
        syns_posts = torch.stack(syns_posts, dim = 4)
        ctx.save_for_backward(mem_updates, outputs, mems, syns_posts, torch.tensor([threshold, tau_s, theta_m])) #used for backprop

        exit()

        return syns_posts


    @staticmethod
    def backward(ctx, grad_delta):
        (delta_u, outputs, u, syns, others) = ctx.saved_tensors 
        shape = grad_delta.shape
        n_steps = shape[4] 
        threshold = others[0].item() 
        tau_s = others[1].item()
        theta_m = others[2].item() #ties into Vmem graph and how it decays
        # print(grad_delta.shape)

        th = 1/(4 * tau_s)

        grad = torch.zeros_like(grad_delta)

        syn_a = glv.syn_a.repeat(shape[0], shape[1], shape[2], shape[3], 1)
        partial_a = glv.syn_a/(-tau_s)
        partial_a = partial_a.repeat(shape[0], shape[1], shape[2], shape[3], 1)

        o = torch.zeros(shape[0], shape[1], shape[2], shape[3]).cuda()


        theta = torch.zeros(shape[0], shape[1], shape[2], shape[3]).cuda()
        for t in range(n_steps-1, -1, -1): #adjusted for indexing from n_steps-1 to 0
            time_end = n_steps 
            time_len = time_end-t #this will be less impactful for spikes near time_end

            count = time_len
            sum = 0
            while (count > 0):
                sum += glv.syn_a[..., count-1]
                count -= 1
            alpha = (1 / sum)
            print(t, alpha)

            out = outputs[..., t]

            partial_u = torch.clamp(-1/delta_u[..., t], -8, 0) * out #constant no matter where spiking occurs 

            #OG:
            #partial_a_partial_u = partial_u.unsqueeze(-1).repeat(1, 1, 1, 1, time_len) * partial_a[..., 0:time_len] 

            #NEW:
            #print(alpha.shape, partial_a[..., 0:time_len].shape, partial_u.unsqueeze(-1).shape)
            partial_a_partial_u = partial_u.unsqueeze(-1).repeat(1, 1, 1, 1, time_len) * alpha * partial_a[..., 0:time_len]

            grad_tmp = torch.sum(partial_a_partial_u*grad_delta[..., t:time_end]*tau_s, dim=4) 

            if t!=n_steps-1:
                grad_tmp += theta * u[..., t] * (-1) * theta_m * partial_u
                grad_tmp += theta * (1-theta_m) * (1-out)
          
            theta = grad_tmp * out + theta * (1-out) * (1-theta_m)

            grad_a = torch.sum(syn_a[..., 0:time_len]*grad_delta[..., t:time_end], dim=-1)

            a = 0.2
            f = torch.clamp((-1 * u[..., t] + threshold) / a, -8, 8)
            f = torch.exp(f)
            f = f / ((1 + f) * (1 + f) * a)

            grad_a = grad_a * f

            syn = syns[..., t]

            grad_tmp[syn<th] = grad_a[syn<th]

            grad[..., t] = grad_tmp

        #exit()

        return grad, None, None
    
            
                




    
