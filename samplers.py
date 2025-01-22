import torch
import math


def sample(n, rho=1, scale=8, T=0.5, start=0, stop=1, dt=0.005, randgrid = 30, thin=0, device=torch.device('cuda:0'), dtype=torch.float64):
        l = math.ceil((stop - start) / dt) + 1
        
        ts = torch.linspace(0, 1, l, device=device, dtype=dtype)
        
        b1 = math.sqrt(2 / T) * torch.sin(4 * torch.pi * ts / T)
        b2 = math.sqrt(2 / T) * torch.cos(6 * torch.pi * ts / T)
        c = torch.randn((n, 3), dtype=dtype, device=device)
       
        b1 = (b1 * c[:, 1].view(n, 1))
        b2 = (b2 * c[:, 2].view(n, 1))
        
        x = (b1 + b2 + c[:, 0].view(n, 1))
       
        c1 = (0.75 - 0.25) * torch.rand((n, 2), device=device, dtype=dtype) + 0.25
        
        i1 = (ts - c1[:,0].view(n, 1, 1)).view(n, -1)**2
        i2 = (ts - c1[:,1].view(n, 1, 1)).view(n, -1)**2
        
        zero_column = torch.zeros(n, 1, device=device, dtype=dtype)
        x1 = torch.cat((zero_column, (x[:,range(l-1)] * dt).cumsum(dim=1)), dim=1) * i2
        x2 = torch.cat((zero_column, (x[:,range(l-1)] * i1[:,range(l-1)] * dt).cumsum(dim=1)), dim=1)
        
        y = rho * (x2 - x1) * scale
        y = y
        x = x
        
        return x.view(n, l, 1)[:, range(0, l, 2**thin),:], y.view(n, l, 1)[:, range(0, l, 2**thin),:]
    
def generate(batch_size, length, dimension, device = torch.device('cpu')):
  random_walks = torch.randn(batch_size, length, dimension, dtype = torch.double, device = device) / math.sqrt(length)
  start = torch.zeros([batch_size, 1, dimension], device=device, dtype=torch.double)
  random_walks = torch.cat((start, random_walks), dim=1)
  random_walks = torch.cumsum(random_walks, dim=1)
  return random_walks

def add_time(x, start=0, stop=1):
    device = x.device
    dtype = x.dtype

    l = x.shape[1]

    t = torch.linspace(start, stop, l, device=device, dtype=dtype)
    t = t.unsqueeze(0).unsqueeze(-1)
    return torch.cat((x, t.expand(x.shape[0], x.shape[1], 1)), dim=-1)

def std_norm(x):
    return 2 - 1 / (1 + x.log())

def time_norm(x, time=True):
    if time:
        x = add_time(x)
    return x / x.pow(2).sum(dim=2).sqrt().max(dim=1).values.view(x.shape[0], 1, 1)


def variation(x):
    diffs = x[:, 1:, :] - x[:, :-1, :]
    euclidean_norms = torch.norm(diffs, p=2, dim=-1)
    return euclidean_norms.sum(dim=-1)

def var_norm(x):
    return x / variation(x).view(x.shape[0], 1, 1).sqrt()

def norm(x):
    return x.pow(2).sum(dim=2).sqrt().max(dim=1).values

class NonLinearSDE():
    def __init__(self):
        return
    
    @staticmethod
    def sample(n, theta=0, dt=0.005, start=0, stop=1, thin=0, device=torch.device('cuda:0'), dtype=torch.float64):
        # Initialize tensors for the results
        num_steps = math.ceil((stop - start) / dt)
        X = torch.zeros(n, num_steps + 1, device=device, dtype=dtype)
        Y = torch.zeros(n, num_steps + 1, device=device, dtype=dtype)
        X[0], Y[0] = 0, 0

        # Wiener processes for the batch
        W1 = math.sqrt(dt) * torch.randn(n, num_steps, device=device, dtype=dtype)
        W2 = math.sqrt(dt) * torch.randn(n, num_steps, device=device, dtype=dtype)

        for t in range(1, num_steps + 1):
            drift_X = -X[:,t - 1]**3
            diffusion_X = torch.sqrt(1 + X[:,t - 1] ** 2)

            drift_Y = theta * torch.sin(X[:,t - 1]) - Y[:,t - 1]
            diffusion_Y = math.sqrt(theta) * torch.exp(-X[:,t - 1] ** 2) + 0.5

            X[:,t] = X[:,t - 1] + drift_X * dt + diffusion_X * W1[:,t - 1]
            Y[:,t] = Y[:,t - 1] + drift_Y * dt + diffusion_Y * W2[:,t - 1]

        return X.view(n, num_steps + 1, 1)[:,range(0, num_steps + 1, 2**thin),:], Y.view(n, num_steps + 1, -1)[:,range(0, num_steps + 1, 2**thin),:]