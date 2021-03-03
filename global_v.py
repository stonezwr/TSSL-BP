import torch


n_steps = None
syn_a = None
tau_s = None

def init(n_t, ts):   
    global n_steps, syn_a, partial_a, tau_s
    n_steps = n_t
    tau_s = ts
    syn_a = torch.zeros(1, 1, 1, 1, n_steps).cuda()
    syn_a[..., 0] = 1
    for t in range(n_steps-1):
        syn_a[..., t+1] = syn_a[..., t] - syn_a[..., t] / tau_s 
    syn_a /= tau_s
