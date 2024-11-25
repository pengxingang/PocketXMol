import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


class TimeEmbedder(nn.Module):
    def __init__(self, dim, total_time):
        super().__init__()
        self.dim = dim
        self.total_time = total_time

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        rel_t = x / (self.total_time - 1)
        emb_sin  = [torch.sin((rel_t + bias) * 0.5 * np.pi) for bias in torch.linspace(0, 1, half_dim+1, device=device)[:-1]]
        emb_cos = [torch.cos((rel_t + bias) * 0.5 * np.pi) for bias in torch.linspace(0, 1, self.dim - half_dim+1, device=device)[:-1]]
        emb = torch.stack(emb_sin + emb_cos, dim=-1)
        return emb

class SineTimeEmbedder(nn.Module):
    def __init__(self, dim, num_steps, rescale_steps=5000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


## --- torch utils ---
def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x

def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)


## --- probabily ---

# categorical diffusion related
def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def extract(coef, t, batch, ndim=2):
    out = coef[t][batch]
    # warning: test wrong code!
    # out = coef[batch]
    # return out.view(-1, *((1,) * (len(out_shape) - 1)))
    if ndim == 1:
        return out
    elif ndim == 2:
        return out.unsqueeze(-1)
    elif ndim == 3:
        return out.unsqueeze(-1).unsqueeze(-1)
    else:
        raise NotImplementedError('ndim > 3')

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits).argmax(dim=-1)
    # sample_onehot = F.one_hot(sample, self.num_classes)
    # log_sample = index_to_log_onehot(sample, self.num_classes)
    return sample_index

def categorical_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=-1)
    return kl

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=-1)


# ----- beta  schedule -----

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


def advance_schedule(timesteps, scale_start, scale_end, width, return_alphas_bar=False):
    k = width
    A0 = scale_end
    A1 = scale_start

    a = (A0-A1)/(sigmoid(-k) - sigmoid(k))
    b = 0.5 * (A0 + A1 - a)

    x = np.linspace(-1, 1, timesteps)
    y = a * sigmoid(- k * x) + b
    # print(y)
    
    alphas_cumprod = y 
    alphas = np.zeros_like(alphas_cumprod)
    alphas[0] = alphas_cumprod[0]
    alphas[1:] = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    betas = np.clip(betas, 0, 1)
    if not return_alphas_bar:
        return betas
    else:
        return betas, alphas_cumprod

def segment_schedule(timesteps, time_segment, segment_diff):
    assert np.sum(time_segment) == timesteps
    alphas_cumprod = []
    for i in range(len(time_segment)):
        time_this = time_segment[i] + 1
        params = segment_diff[i]
        _, alphas_this = advance_schedule(time_this, **params, return_alphas_bar=True)
        alphas_cumprod.extend(alphas_this[1:])
    alphas_cumprod = np.array(alphas_cumprod)
    
    alphas = np.zeros_like(alphas_cumprod)
    alphas[0] = alphas_cumprod[0]
    alphas[1:] = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    betas = np.clip(betas, 0, 1)
    return betas

def sigmoid(x):
    return 1 / (np.exp(-x) + 1)

def get_beta_schedule(beta_schedule, num_timesteps, **kwargs):
    
    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    kwargs['beta_start'] ** 0.5,
                    kwargs['beta_end'] ** 0.5,
                    num_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            kwargs['beta_start'], kwargs['beta_end'], num_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = kwargs['beta_end'] * np.ones(num_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_timesteps, 1, num_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        s = dict.get(kwargs, 's', 6)
        betas = np.linspace(-s, s, num_timesteps)
        betas = sigmoid(betas) * (kwargs['beta_end'] - kwargs['beta_start']) + kwargs['beta_start']
    elif beta_schedule == "cosine":
        s = dict.get(kwargs, 's', 0.008)
        betas = cosine_beta_schedule(num_timesteps, s=s)
    elif beta_schedule == "advance":
        scale_start = dict.get(kwargs, 'scale_start', 0.999)
        scale_end = dict.get(kwargs, 'scale_end', 0.001)
        width = dict.get(kwargs, 'width', 2)
        betas = advance_schedule(num_timesteps, scale_start, scale_end, width)
    elif beta_schedule == "segment":
        betas = segment_schedule(num_timesteps, kwargs['time_segment'], kwargs['segment_diff'])
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_timesteps,)
    return betas


# def get_segment_schedule(time_intervals, segment_diff):
#     t0 = 0
#     betas = []
#     for i, t1 in enumerate(time_intervals):
#         betas_seg = advance_schedule(t1 - t0, **segment_diff[i])
#         betas.extend(betas_seg)
#         t0 = t1
#     return np.array(betas)