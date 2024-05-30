import torch


n_steps = None
syn_a = None
tau_s = None

def init(n_t, ts):   
    global n_steps, syn_a, partial_a, tau_s
    n_steps = n_t
    tau_s = ts
    #alter this to be able to convert to continous time
    syn_a = torch.zeros(1, 1, 1, 1, n_steps).cuda() # 5 time steps = 1x5 vector
    syn_a[..., 0] = 1 #base condition
    for t in range(n_steps-1):
        syn_a[..., t+1] = syn_a[..., t] - syn_a[..., t] / tau_s #tau_s is your decay function, so next array element is:
                                                                #previous element minus prev with decay
        
    syn_a /= tau_s
    print(syn_a)
