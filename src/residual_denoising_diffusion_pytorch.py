from __future__ import division
import glob
import math
import random
from collections import namedtuple
from functools import partial
from pathlib import Path
import numpy as np
import torch

np.bool = np.bool_
import cv2
from accelerate import Accelerator
from datasets.get_dataset import dataset
from einops import rearrange, reduce
from ema_pytorch import EMA
from torch import einsum
from torch.optim import RAdam
from torchvision import utils
from tqdm.auto import tqdm
from ssim2 import SSIM
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt
import pytorch_fid_wrapper as pfw
from resnet import *
from torchvision import transforms
import ssim_loss
import mxnet as mx
from mxnet import gluon
import torch.utils.dlpack as tdl
from src.models import stainNorm_Vahadane, stainNorm_Reinhard, stainNorm_Macenko


ModelResPrediction = namedtuple(
    'ModelResPrediction', ['pred_res', 'pred_noise', 'pred_x_start'])
# helpers functions


def set_seed(SEED):
    # initialize random seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

def normal_auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# normalization functions


def normalize_to_neg_one_to_one(img):
    if isinstance(img, list):
        return [img[k] * 2 - 1 for k in range(len(img))]
    else:
        return img * 2 - 1


def unnormalize_to_zero_to_one(img):
    if isinstance(img, list):
        return [(img[k] + 1) * 0.5 for k in range(len(img))]
    else:
        return (img + 1) * 0.5

# small helper modules


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1',
                     partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(
            half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        condition=False,
        input_condition=False,
        img_to_img_translation=False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels + channels * \
            (1 if self_condition else 0) + channels * \
            (1 if condition and (not img_to_img_translation) else 0) + channels * (1 if input_condition else 0)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
                    dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                    dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
        x = x.float()
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


class UnetRes(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        num_unet=1,
        condition=False,
        input_condition=False,
        objective='pred_res_noise',
        test_res_or_noise="res_noise",
        img_to_img_translation=False
    ):
        super().__init__()
        self.condition = condition
        self.input_condition = input_condition
        self.channels = channels
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        self.self_condition = self_condition
        self.num_unet = num_unet
        self.objective = objective
        self.test_res_or_noise = test_res_or_noise
        self.img_to_img_translation = img_to_img_translation
        # determine dimensions
        if self.num_unet == 2:
            self.unet0 = Unet(dim,
                              init_dim=init_dim,
                              out_dim=out_dim,
                              dim_mults=dim_mults,
                              channels=channels,
                              self_condition=self_condition,
                              resnet_block_groups=resnet_block_groups,
                              learned_variance=learned_variance,
                              learned_sinusoidal_cond=learned_sinusoidal_cond,
                              random_fourier_features=random_fourier_features,
                              learned_sinusoidal_dim=learned_sinusoidal_dim,
                              condition=condition,
                              input_condition=input_condition,
                              img_to_img_translation=img_to_img_translation)
            self.unet1 = Unet(dim,
                              init_dim=init_dim,
                              out_dim=out_dim,
                              dim_mults=dim_mults,
                              channels=channels,
                              self_condition=self_condition,
                              resnet_block_groups=resnet_block_groups,
                              learned_variance=learned_variance,
                              learned_sinusoidal_cond=learned_sinusoidal_cond,
                              random_fourier_features=random_fourier_features,
                              learned_sinusoidal_dim=learned_sinusoidal_dim,
                              condition=condition,
                              input_condition=input_condition,
                              img_to_img_translation=img_to_img_translation)
        elif self.num_unet == 1:
            self.unet0 = Unet(dim,
                              init_dim=init_dim,
                              out_dim=out_dim,
                              dim_mults=dim_mults,
                              channels=channels,
                              self_condition=self_condition,
                              resnet_block_groups=resnet_block_groups,
                              learned_variance=learned_variance,
                              learned_sinusoidal_cond=learned_sinusoidal_cond,
                              random_fourier_features=random_fourier_features,
                              learned_sinusoidal_dim=learned_sinusoidal_dim,
                              condition=condition,
                              input_condition=input_condition,
                              img_to_img_translation=img_to_img_translation)

    def forward(self, x, time, x_self_cond=None):
        if self.num_unet == 2:
            if self.test_res_or_noise == "res_noise":
                return self.unet0(x, time[0], x_self_cond=x_self_cond), self.unet1(x, time[1], x_self_cond=x_self_cond)
            elif self.test_res_or_noise == "res":
                return self.unet0(x, time[0], x_self_cond=x_self_cond), 0
            elif self.test_res_or_noise == "noise":
                return 0, self.unet1(x, time[1], x_self_cond=x_self_cond)
            if self.test_res_or_noise == "x0_noise":
                return self.unet0(x, time[0], x_self_cond=x_self_cond), self.unet1(x, time[1], x_self_cond=x_self_cond)
            elif self.test_res_or_noise == "x0":
                return self.unet0(x, time[0], x_self_cond=x_self_cond), 0
            elif self.test_res_or_noise == "noise":
                return 0, self.unet1(x, time[1], x_self_cond=x_self_cond)
        elif self.num_unet == 1:
            if self.objective == 'pred_res_noise':
                # num_unet=2
                pass
            elif self.objective == 'pred_x0_noise':
                # num_unet=2
                pass
            elif self.objective == "pred_noise":
                time = time[1]
            elif self.objective == "pred_res":
                time = time[0]
            elif self.objective == "pred_x0":
                time = time[0]
            return [self.unet0(x, time, x_self_cond=x_self_cond)]

# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def gen_coefficients(timesteps, schedule="increased", sum_scale=1, ratio=1):
    if schedule == "increased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        alphas = y/y_sum
    elif schedule == "decreased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        y = torch.flip(y, dims=[0])
        alphas = y/y_sum
    elif schedule == "average":
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float32)
    elif schedule == "normal":
        sigma = 1.0
        mu = 0.0
        x = np.linspace(-3+mu, 3+mu, timesteps, dtype=np.float32)
        y = np.e**(-((x-mu)**2)/(2*(sigma**2)))/(np.sqrt(2*np.pi)*(sigma**2))
        y = torch.from_numpy(y)
        alphas = y/y.sum()
    else:
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float32)
    assert (alphas.sum()-1).abs() < 1e-6

    return alphas*sum_scale

# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999) -> torch.Tensor:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class ResidualDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        num_samples = 1,
        sampling_timesteps=None,
        loss_type='l1',
        objective='pred_res_noise',
        ddim_sampling_eta=0.,
        condition=False,
        sum_scale=None,
        input_condition=False,
        input_condition_mask=False,
        test_res_or_noise="None",
        img_to_img_translation=False
    ):
        super().__init__()
        assert not (
            type(self) == ResidualDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond
        self.num_samples = num_samples
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.image_size = image_size
        self.objective = objective
        self.condition = condition
        self.input_condition = input_condition
        self.input_condition_mask = input_condition_mask
        self.test_res_or_noise = test_res_or_noise
        self.img_to_img_translation = img_to_img_translation

        if self.condition:
            self.sum_scale = sum_scale if sum_scale else 0.01
            ddim_sampling_eta = 0.
        else:
            self.sum_scale = sum_scale if sum_scale else 1.

        convert_to_ddim = True
        if convert_to_ddim:
            beta_schedule = "linear"
            beta_start = 0.0001
            beta_end = 0.02
            if beta_schedule == "linear":
                betas = torch.linspace(
                    beta_start, beta_end, timesteps, dtype=torch.float32)
            elif beta_schedule == "scaled_linear":
                # this schedule is very specific to the latent diffusion model.
                betas = (
                    torch.linspace(beta_start**0.5, beta_end**0.5,
                                   timesteps, dtype=torch.float32) ** 2
                )
            elif beta_schedule == "squaredcos_cap_v2":
                # Glide cosine schedule
                betas = betas_for_alpha_bar(timesteps)
            else:
                raise NotImplementedError(
                    f"{beta_schedule} does is not implemented for {self.__class__}")

            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            alphas_cumsum = 1-alphas_cumprod ** 0.5
            betas2_cumsum = 1-alphas_cumprod

            alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
            betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
            alphas = alphas_cumsum-alphas_cumsum_prev
            alphas[0] = 0
            betas2 = betas2_cumsum-betas2_cumsum_prev
            betas2[0] = 0
        else:
            alphas = gen_coefficients(timesteps, schedule="decreased")
            betas2 = gen_coefficients(
                timesteps, schedule="increased", sum_scale=self.sum_scale)

            alphas_cumsum = alphas.cumsum(dim=0).clip(0, 1)
            betas2_cumsum = betas2.cumsum(dim=0).clip(0, 1)

            alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
            betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)

        betas_cumsum = torch.sqrt(betas2_cumsum)
        posterior_variance = betas2*betas2_cumsum_prev/betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters
        # default num sampling timesteps to number of timesteps at training
        self.sampling_timesteps = default(sampling_timesteps, timesteps)

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('alphas', alphas)
        register_buffer('alphas_cumsum', alphas_cumsum)
        register_buffer('one_minus_alphas_cumsum', 1-alphas_cumsum)
        register_buffer('betas2', betas2)
        register_buffer('betas', torch.sqrt(betas2))
        register_buffer('betas2_cumsum', betas2_cumsum)
        register_buffer('betas_cumsum', betas_cumsum)
        register_buffer('posterior_mean_coef1',
                        betas2_cumsum_prev/betas2_cumsum)
        register_buffer('posterior_mean_coef2', (betas2 *
                        alphas_cumsum_prev-betas2_cumsum_prev*alphas)/betas2_cumsum)
        register_buffer('posterior_mean_coef3', betas2/betas2_cumsum)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6

    def init(self):
        timesteps = 1000

        convert_to_ddim = True
        if convert_to_ddim:
            beta_schedule = "linear"
            beta_start = 0.0001
            beta_end = 0.02
            if beta_schedule == "linear":
                betas = torch.linspace(
                    beta_start, beta_end, timesteps, dtype=torch.float32)
            elif beta_schedule == "scaled_linear":
                # this schedule is very specific to the latent diffusion model.
                betas = (
                    torch.linspace(beta_start**0.5, beta_end**0.5,
                                   timesteps, dtype=torch.float32) ** 2
                )
            elif beta_schedule == "squaredcos_cap_v2":
                # Glide cosine schedule
                betas = betas_for_alpha_bar(timesteps)
            else:
                raise NotImplementedError(
                    f"{beta_schedule} does is not implemented for {self.__class__}")

            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            alphas_cumsum = 1-alphas_cumprod ** 0.5
            betas2_cumsum = 1-alphas_cumprod

            alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
            betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
            alphas = alphas_cumsum-alphas_cumsum_prev
            alphas[0] = alphas[1]
            betas2 = betas2_cumsum-betas2_cumsum_prev
            betas2[0] = betas2[1]

            # adjust
            # alphas = gen_coefficients(timesteps, schedule="average", ratio=0)
            # alphas_cumsum = alphas.cumsum(dim=0).clip(0, 1)
            # alphas_cumsum_prev = F.pad(
            #     alphas_cumsum[:-1], (1, 0), value=alphas_cumsum[1])

            # betas2 = gen_coefficients(
            #     timesteps, schedule="average", sum_scale=self.sum_scale, ratio=0)
            # betas2_cumsum = betas2.cumsum(dim=0).clip(0, 1)
            # betas2_cumsum_prev = F.pad(
            #     betas2_cumsum[:-1], (1, 0), value=betas2_cumsum[1])
        else:
            alphas = gen_coefficients(timesteps, schedule="average", ratio=1)
            betas2 = gen_coefficients(
                timesteps, schedule="increased", sum_scale=self.sum_scale, ratio=3)

            alphas_cumsum = alphas.cumsum(dim=0).clip(0, 1)
            betas2_cumsum = betas2.cumsum(dim=0).clip(0, 1)

            alphas_cumsum_prev = F.pad(
                alphas_cumsum[:-1], (1, 0), value=alphas_cumsum[1])
            betas2_cumsum_prev = F.pad(
                betas2_cumsum[:-1], (1, 0), value=betas2_cumsum[1])

        betas_cumsum = torch.sqrt(betas2_cumsum)
        posterior_variance = betas2*betas2_cumsum_prev/betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)

        self.alphas = alphas
        self.alphas_cumsum = alphas_cumsum
        self.one_minus_alphas_cumsum = 1-alphas_cumsum
        self.betas2 = betas2
        self.betas = torch.sqrt(betas2)
        self.betas2_cumsum = betas2_cumsum
        self.betas_cumsum = betas_cumsum
        self.posterior_mean_coef1 = betas2_cumsum_prev/betas2_cumsum
        self.posterior_mean_coef2 = (
            betas2 * alphas_cumsum_prev-betas2_cumsum_prev*alphas)/betas2_cumsum
        self.posterior_mean_coef3 = betas2/betas2_cumsum
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(
            posterior_variance.clamp(min=1e-20))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6

    def predict_noise_from_res(self, x_t, t, x_input, pred_res):
        return (
            (x_t-x_input-(extract(self.alphas_cumsum, t, x_t.shape)-1)
             * pred_res)/extract(self.betas_cumsum, t, x_t.shape)
        )

    def predict_start_from_xinput_noise(self, x_t, t, x_input, noise):
        return (
            (x_t-extract(self.alphas_cumsum, t, x_t.shape)*x_input -
             extract(self.betas_cumsum, t, x_t.shape) * noise)/extract(self.one_minus_alphas_cumsum, t, x_t.shape)
        )

    def predict_start_from_res_noise(self, x_t, t, x_res, noise):
        return (
            x_t-extract(self.alphas_cumsum, t, x_t.shape) * x_res -
            extract(self.betas_cumsum, t, x_t.shape) * noise
        )

    def q_posterior_from_res_noise(self, x_res, noise, x_t, t):
        return (x_t-extract(self.alphas, t, x_t.shape) * x_res -
                (extract(self.betas2, t, x_t.shape)/extract(self.betas_cumsum, t, x_t.shape)) * noise)

    def q_posterior(self, pred_res, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_t +
            extract(self.posterior_mean_coef2, t, x_t.shape) * pred_res +
            extract(self.posterior_mean_coef3, t, x_t.shape) * x_start
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_input, x, t, x_input_condition=0, x_self_cond=None, clip_denoised=True):
        if not self.condition:
            x_in = x
        else:
            if self.img_to_img_translation:
                if self.input_condition:
                    x_in = torch.cat((x, x_input_condition), dim=1)
                else:
                    x_in = x
            else:
                if self.input_condition:
                    x_in = torch.cat((x, x_input, x_input_condition), dim=1)
                else:
                    x_in = torch.cat((x, x_input), dim=1)
        model_output = self.model(x_in,
                                  [self.alphas_cumsum[t]*self.num_timesteps,
                                      self.betas_cumsum[t]*self.num_timesteps],
                                  x_self_cond)
        maybe_clip = partial(torch.clamp, min=-1.,
                             max=1.) if clip_denoised else identity

        if self.objective == 'pred_res_noise':
            if self.test_res_or_noise == "res_noise":
                pred_res = model_output[0]
                pred_noise = model_output[1]
                pred_res = maybe_clip(pred_res)
                x_start = self.predict_start_from_res_noise(
                    x, t, pred_res, pred_noise)
                x_start = maybe_clip(x_start)
            elif self.test_res_or_noise == "res":
                pred_res = model_output[0]
                pred_res = maybe_clip(pred_res)
                pred_noise = self.predict_noise_from_res(
                    x, t, x_input, pred_res)
                x_start = x_input - pred_res
                x_start = maybe_clip(x_start)
            elif self.test_res_or_noise == "noise":
                pred_noise = model_output[1]
                x_start = self.predict_start_from_xinput_noise(
                    x, t, x_input, pred_noise)
                x_start = maybe_clip(x_start)
                pred_res = x_input - x_start
                pred_res = maybe_clip(pred_res)
        elif self.objective == 'pred_x0_noise':
            if self.test_res_or_noise == "x0_noise":
                pred_res = x_input-model_output[0]
                pred_noise = model_output[1]
                pred_res = maybe_clip(pred_res)
                x_start = maybe_clip(model_output[0])
            elif self.test_res_or_noise == "x0":
                pred_res = x_input-model_output[0]
                pred_res = maybe_clip(pred_res)
                pred_noise = self.predict_noise_from_res(
                    x, t, x_input, pred_res)
                x_start = maybe_clip(model_output[0])
            elif self.test_res_or_noise == "noise":
                pred_noise = model_output[1]
                x_start = self.predict_start_from_xinput_noise(
                    x, t, x_input, pred_noise)
                x_start = maybe_clip(x_start)
                pred_res = x_input - x_start
                pred_res = maybe_clip(pred_res)
        elif self.objective == "pred_noise":
            pred_noise = model_output[0]
            x_start = self.predict_start_from_xinput_noise(
                x, t, x_input, pred_noise)
            x_start = maybe_clip(x_start)
            pred_res = x_input - x_start
            pred_res = maybe_clip(pred_res)
        elif self.objective == "pred_res":
            pred_res = model_output[0]
            pred_res = maybe_clip(pred_res)
            pred_noise = self.predict_noise_from_res(x, t, x_input, pred_res)
            x_start = x_input - pred_res
            x_start = maybe_clip(x_start)
        elif self.objective == "pred_x0":
            pred_res = x_input-model_output[0]
            pred_res = maybe_clip(pred_res)
            pred_noise = self.predict_noise_from_res(x, t, x_input, pred_res)
            x_start = x_input - pred_res
            x_start = maybe_clip(x_start)

        return ModelResPrediction(pred_res, pred_noise, x_start)

    def p_mean_variance(self, x_input, x, t, x_input_condition=0, x_self_cond=None):
        preds = self.model_predictions(
            x_input, x, t, x_input_condition, x_self_cond)
        pred_res = preds.pred_res
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            pred_res=pred_res, x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x_input, x, t: int, x_input_condition=0, x_self_cond=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full(
            (x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x_input, x=x, t=batched_times, x_input_condition=x_input_condition, x_self_cond=x_self_cond)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, x_input, shape, last=True):
        if self.input_condition:
            x_input_condition = x_input[1]
        else:
            x_input_condition = 0
        x_input = x_input[0]

        batch, device = shape[0], self.betas.device

        if self.condition:
            img = x_input+math.sqrt(self.sum_scale) * \
                torch.randn(shape, device=device)
            input_add_noise = img
        else:
            img = torch.randn(shape, device=device)

        x_start = None

        if not last:
            img_list = []

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(
                x_input, img, t, x_input_condition, self_cond)

            if not last:
                img_list.append(img)

        if self.condition:
            if not last:
                img_list = [input_add_noise]+img_list
            else:
                img_list = [input_add_noise, img]
            return unnormalize_to_zero_to_one(img_list)
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)



    @torch.no_grad()
    def ddim_sample(self, x_input, shape,file_name=None,last=True,XAI=False):
        if self.input_condition:
            x_input_condition = x_input[1]
        else:
            x_input_condition = 0
        x_input = x_input[0]

        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        if self.condition:
            img = x_input+math.sqrt(self.sum_scale) * \
                torch.randn(shape, device=device)
            input_add_noise = img
        else:
            img = torch.randn(shape, device=device)

        x_start = None
        type = "use_pred_noise"

        if not last:
            img_list = []

        eta = 0
        heatmap_list_noise = []
        auc_list_noise = []
        auc_noise = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full(
                (batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            preds = self.model_predictions(
                x_input, img, time_cond, x_input_condition, self_cond)

            pred_res = preds.pred_res
            pred_noise = preds.pred_noise
            x_start = preds.pred_x_start

            if time_next < 0:
                img = x_start
                if not last:
                    img_list.append(img)
                continue

            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            alpha = alpha_cumsum-alpha_cumsum_next

            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2 = betas2_cumsum-betas2_cumsum_next
            # betas2 = 1-(1-betas2_cumsum)/(1-betas2_cumsum_next)
            betas = betas2.sqrt()
            betas_cumsum = self.betas_cumsum[time]
            betas_cumsum_next = self.betas_cumsum[time_next]
            sigma2 = eta * (betas2*betas2_cumsum_next/betas2_cumsum)
            sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum = (
                betas2_cumsum_next-sigma2).sqrt()/betas_cumsum

            if eta == 0:
                noise = 0
            else:
                noise = torch.randn_like(img)

            input_latent = copy.deepcopy(img)
            pred_res = 0
            if type == "use_pred_noise":
                img = img - alpha*pred_res - \
                    (betas_cumsum-(betas2_cumsum_next-sigma2).sqrt()) * \
                    pred_noise + sigma2.sqrt()*noise
            elif type == "use_x_start":
                img = sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum*img + \
                    (1-sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum)*x_start + \
                    (alpha_cumsum_next-alpha_cumsum*sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum)*pred_res + \
                    sigma2.sqrt()*noise
            elif type == "special_eta_0":
                img = img - alpha*pred_res - \
                    (betas_cumsum-betas_cumsum_next)*pred_noise
            elif type == "special_eta_1":
                img = img - alpha*pred_res - betas2/betas_cumsum*pred_noise + \
                    betas*betas2_cumsum_next.sqrt()/betas_cumsum*noise



            if not last:
                img_list.append(img)
                #print(len(img_list))
            img_heat = copy.deepcopy(img)

            if XAI:


                heatmap_noise, auc_noise = self.generate_saliency_map(img_heat, input_latent, x_input, time_cond,
                                                                      x_input_condition, self_cond, mode='all', sim_func='ssim' ,
                                                                      prob_thresh=0.5, alpha_res= alpha,beta_noise=(betas_cumsum - (betas2_cumsum_next - sigma2).sqrt()), noise_sal=sigma2.sqrt() * noise,get_auc_score=True,auc_mode='ins',file_name=file_name)
                print("auc_noise:{}".format(auc_noise))
                heatmap_noise = F.relu(heatmap_noise)
                heatmap_noise_min, heatmap_noise_max = heatmap_noise.min(), heatmap_noise.max()
                heatmap_noise = (heatmap_noise - heatmap_noise_min) / (heatmap_noise_max - heatmap_noise_min)
                heatmap_noise = heatmap_noise.cpu().data
                heatmap_noise = (heatmap_noise - heatmap_noise.min()).div(heatmap_noise.max() - heatmap_noise.min()).data
                heatmap_noise = cv2.applyColorMap(np.uint8(255 * heatmap_noise.float()), cv2.COLORMAP_JET)
                heatmap_noise = torch.from_numpy(heatmap_noise).permute(2, 0, 1).float().div(255)
                heatmap_noise = ((heatmap_noise - heatmap_noise.min()) / (heatmap_noise.max() - heatmap_noise.min())).unsqueeze(0)
                #heatmap_noise = F.interpolate(heatmap_noise, size=(512, 512), mode='bicubic', align_corners=False)
                heatmap_list_noise.append(heatmap_noise)

                auc_list_noise.append(auc_noise)

        img_final_heat = copy.deepcopy(img)
        heatmap_final = None
        """
        heatmap_final, auc_final = self.generate_saliency_map2(img_final_heat, input_latent, x_input, time_cond,

                                                                  x_input_condition, self_cond, mode='all', sim_func='ssim' ,
                                                                    prob_thresh=0.5, alpha_res= alpha,beta_noise=(betas_cumsum - (betas2_cumsum_next - sigma2).sqrt()), noise_sal=sigma2.sqrt() * noise)
        """
        if self.condition:
            if not last:
                img_list = [input_add_noise]+img_list
            else:
                img_list = [input_add_noise, img]
            return unnormalize_to_zero_to_one(img_list), heatmap_list_noise,auc_noise, heatmap_final
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)



    @torch.no_grad()
    def sample(self, x_input=0, batch_size=16, last=True, file_name=None,xai=False):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        if self.condition:
            if self.input_condition and self.input_condition_mask:
                x_input[0] = normalize_to_neg_one_to_one(x_input[0])
            else:
                x_input = normalize_to_neg_one_to_one(x_input)
            batch_size, channels, h, w = x_input[0].shape
            size = (batch_size, channels, h, w)
        else:
            size = (batch_size, channels, image_size, image_size)
        return sample_fn(x_input, size, file_name = file_name, last=last,XAI=xai)

    def q_sample(self, x_start, x_res, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            x_start+extract(self.alphas_cumsum, t, x_start.shape) * x_res +
            extract(self.betas_cumsum, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, imgs, t, noise=None):
        if isinstance(imgs, list):  # Condition
            if self.input_condition:
                x_input_condition = imgs[2]
            else:
                x_input_condition = 0
            x_input = imgs[1]
            x_start = imgs[0]  # gt = imgs[0], input = imgs[1]
        else:  # Generation
            x_input = 0
            x_start = imgs

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_res = x_input - x_start

        b, c, h, w = x_start.shape

        # noise sample
        x = self.q_sample(x_start, x_res, t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        x_self_cond = None
        if self.self_condition and random.random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(
                    x_input, x, t, x_input_condition if self.input_condition else 0).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
        if not self.condition:
            x_in = x
        else:
            if self.img_to_img_translation:
                if self.input_condition:
                    x_in = torch.cat((x, x_input_condition), dim=1)
                else:
                    x_in = x
            else:
                if self.input_condition:
                    x_in = torch.cat((x, x_input, x_input_condition), dim=1)
                else:
                    x_in = torch.cat((x, x_input), dim=1)

        model_out = self.model(x_in,
                               [self.alphas_cumsum[t]*self.num_timesteps,
                                   self.betas_cumsum[t]*self.num_timesteps],
                               x_self_cond)

        target = []
        if self.objective == 'pred_res_noise':
            target.append(x_res)
            target.append(noise)

            pred_res = model_out[0]
            pred_noise = model_out[1]
        elif self.objective == 'pred_x0_noise':
            target.append(x_start)
            target.append(noise)

            pred_res = x_input-model_out[0]
            pred_noise = model_out[1]
        elif self.objective == "pred_noise":
            target.append(noise)

            pred_noise = model_out[0]

        elif self.objective == "pred_res":
            target.append(x_res)

            pred_res = model_out[0]

        elif self.objective == "pred_x0":
            target.append(x_start)

            pred_x0 = model_out[0]

        else:
            raise ValueError(f'unknown objective {self.objective}')

        u_loss = False
        if u_loss:
            x_u = self.q_posterior_from_res_noise(pred_res, pred_noise, x, t)
            u_gt = self.q_posterior_from_res_noise(x_res, noise, x, t)
            loss = 10000*self.loss_fn(x_u, u_gt, reduction='none')
            return [loss]
        else:
            loss_list = []
            ssim_loss_ = ssim_loss.SSIM()
            for i in range(len(model_out)):
                loss = self.loss_fn(model_out[i], target[i], reduction='none')
                loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()
                img1 = model_out[i].cpu().detach().numpy()
                img2 = target[i].cpu().detach().numpy()
                img1 = mx.nd.array(img1)
                img2 = mx.nd.array(img2)

                loss2 = -ssim_loss_(img1, img2)
                #ssim_loss2 = loss2.view(b, -1).mean(dim=-1)
                ssim_loss2 = reduce(loss2, 'b ... -> b (...)', 'mean').mean()
                device_ = loss.device
                ssim_loss2 = torch.from_numpy(ssim_loss2.asnumpy()).to(device_)
                alpha = 0.9
                loss = alpha * loss + (1 - alpha) * ssim_loss2

                loss_list.append(loss)
            return loss_list

    def forward(self, img, *args, **kwargs):
        if isinstance(img, list):
            b, c, h, w, device, img_size, = * \
                img[0].shape, img[0].device, self.image_size
        else:
            b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        if self.input_condition and self.input_condition_mask:
            img[0] = normalize_to_neg_one_to_one(img[0])
            img[1] = normalize_to_neg_one_to_one(img[1])
        else:
            img = normalize_to_neg_one_to_one(img)

        return self.p_losses(img, t, *args, **kwargs)

    @torch.no_grad()
    def generate_saliency_map(
            self,
            target_latents,
            input_latent,
            x_input,
            t,
            x_input_condition,
            self_cond,
            mode,
            rise_num_steps=10,
            hidden_states=None,
            prob_thresh=0.5,
            alpha_res=None,
            beta_noise=None,
            noise_sal=None,
            activation_map=None,
            sim_func="ssim",
            layer_vis=False,
            ssim_mode='structure',
            get_auc_score=False,
            auc_mode='del',
            last=True,
            file_name=None
    ):

        if layer_vis:
            pass
            """
            self.unet.down_blocks[2].resnets[1].register_forward_pre_hook(self.preforward_hook)
            self.activations = dict()
            input_latents = torch.cat([input_latent] * 2).to("cuda")
            target_latents = target_latents
            """

        h, w = target_latents.shape[2:]


        res = torch.zeros((h, w), dtype=torch.float32).to(target_latents.device)

        if sim_func == 'cos':
            score_func = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        elif sim_func == 'ssim':
            score_func = SSIM(15, reduction='none', mode=ssim_mode)

        # for step in tqdm(range(rise_num_steps), desc = "Rise iteration..."):
        # batches = self.num_samples
        for step in range(rise_num_steps):
            if not layer_vis:
                # Activation map version
                if activation_map is not None:
                    pass
                    """
                    self.mask, masked_latents = self.actv_masking_latents(input_latent, prob_thresh=prob_thresh,
                                                                          activ=activation_map)
                    """
                else:
                    # Gaussian random masking version
                    self.mask, masked_latents = self.gau_masking_latents(input_latent, prob_thresh=prob_thresh)

                # masked_latents_input = torch.cat([masked_latents] * 2).to("cuda")
                masked_latents_input = masked_latents.to("cuda")
                # masked_pred = self.unet(masked_latents_input, t, encoder_hidden_states=hidden_states).sample
                preds = self.model_predictions(
                    x_input, masked_latents_input, t, x_input_condition, self_cond)
                pred_res = preds.pred_res
                pred_noise = preds.pred_noise
                x_start = preds.pred_x_start

                if mode == 'res':
                    masked_pred = masked_latents_input - alpha_res * pred_res + noise_sal

                elif mode == 'noise':
                    masked_pred = masked_latents_input - beta_noise * pred_noise + noise_sal
                elif mode == "all":
                    masked_pred = masked_latents_input - alpha_res*pred_res - beta_noise * pred_noise + noise_sal

                mask = self.mask
            else:
                pass
                """
                if activation_map is not None:
                    self.actv_map = activation_map

                masked_pred = self.unet(input_latent, t, encoder_hidden_states=hidden_states).sample
                mask = self.mask.unsqueeze(0).unsqueeze(0)
                mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                """

            # Classifier free guidance
            # masked_noise_pred_uncond, masked_noise_pred_text = masked_pred.chunk(2)
            # masked_noise_pred_cfg = masked_noise_pred_uncond + self.guidance_scale * (
            #            masked_noise_pred_text - masked_noise_pred_uncond)
            masked_noise_pred_cfg = masked_pred

            # #  mask generation
            # masked_latents = self.scheduler.step(masked_noise_pred_cfg, t, masked_latents, **self.extra_step_kwargs).prev_sample
            # pred_img = self.decode_latents(masked_latents)
            # pred = torch.from_numpy(pred_img).permute(0,3,1,2)
            # if step <20:
            #     plt.figure('Mask', figsize=(10,4))
            #     plt.imshow(pred.squeeze(0).permute(1, 2, 0).cpu().numpy())
            #     plt.axis("off")
            #     plt.savefig(os.path.join(f"outputs/mask3/mask_{step}.png"))
            #     # plt.show()

            # #########
            auc_score = None
            # print(target_latents)
            # print(masked_noise_pred_cfg)
            score = score_func(target_latents, masked_noise_pred_cfg).squeeze(0)
            # print(score)

            # score = structural_similarity(target_latents, masked_noise_pred_cfg, multichannel=True,channel_axis=1,data_range=1)
            # print(score)

            res += mask * score
        res = res
        if get_auc_score:
            save_dir = '/mnt/data/result_ge47nej/results_tranlation_test/auc/'
            auc_score = self.auc_run(auc_mode, input_latent, target_latents,x_input,x_input_condition=x_input_condition, self_cond=self_cond, explanation= res,  t = t, encoder_hidden=hidden_states, alpha_res=alpha_res,beta_noise= beta_noise, noise_sal=noise_sal, random=False,verbose=0, save_to=save_dir,file_name=file_name)
        else:
            auc_score = None
        return res, auc_score

    @torch.no_grad()
    def generate_saliency_map2(
            self,
            target_latents,
            input_latent,
            x_input,
            t,
            x_input_condition,
            self_cond,
            mode,
            rise_num_steps=10,
            hidden_states=None,
            prob_thresh=0.5,
            alpha_res=None,
            beta_noise=None,
            noise_sal=None,
            activation_map=None,
            sim_func="ssim",
            layer_vis=False,
            ssim_mode='structure',
            get_auc_score=False,
            auc_mode='ins',
            sample_steps = 10,
            last=True
    ):

        if layer_vis:
            pass
            """
            self.unet.down_blocks[2].resnets[1].register_forward_pre_hook(self.preforward_hook)
            self.activations = dict()
            input_latents = torch.cat([input_latent] * 2).to("cuda")
            target_latents = target_latents
            """

        h, w = target_latents.shape[2:]

        res = torch.zeros((h, w), dtype=torch.float32).to(target_latents.device)

        if sim_func == 'cos':
            score_func = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        elif sim_func == 'ssim':
            score_func = SSIM(15, reduction='none', mode=ssim_mode)

        # for step in tqdm(range(rise_num_steps), desc = "Rise iteration..."):
        # batches = self.num_samples
        for step in range(rise_num_steps):
            if not layer_vis:
                # Activation map version
                if activation_map is not None:
                    pass
                    """
                    self.mask, masked_latents = self.actv_masking_latents(input_latent, prob_thresh=prob_thresh,
                                                                          activ=activation_map)
                    """
                else:
                    # Gaussian random masking version
                    self.mask, masked_latents = self.gau_masking_latents(input_latent, prob_thresh=prob_thresh)
                image_size, channels = self.image_size, self.channels
                #sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample

                batch_size, channels, h, w = x_input.shape
                size = (batch_size, channels, h, w)
                # masked_latents_input = torch.cat([masked_latents] * 2).to("cuda")
                masked_latents_input = masked_latents.to("cuda")
                # masked_pred = self.unet(masked_latents_input, t, encoder_hidden_states=hidden_states).sample

                #preds = self.model_predictions(
                #        x_input, masked_latents_input, t, x_input_condition, self_cond)
                input_add_noise,img = self.ddim_sample2(masked_latents_input, size, last=True)



                mask = self.mask
            else:
                pass
                """
                if activation_map is not None:
                    self.actv_map = activation_map

                masked_pred = self.unet(input_latent, t, encoder_hidden_states=hidden_states).sample
                mask = self.mask.unsqueeze(0).unsqueeze(0)
                mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                """

            # Classifier free guidance
            # masked_noise_pred_uncond, masked_noise_pred_text = masked_pred.chunk(2)
            # masked_noise_pred_cfg = masked_noise_pred_uncond + self.guidance_scale * (
            #            masked_noise_pred_text - masked_noise_pred_uncond)
            masked_noise_pred_cfg = img

            # #  mask generation
            # masked_latents = self.scheduler.step(masked_noise_pred_cfg, t, masked_latents, **self.extra_step_kwargs).prev_sample
            # pred_img = self.decode_latents(masked_latents)
            # pred = torch.from_numpy(pred_img).permute(0,3,1,2)
            # if step <20:
            #     plt.figure('Mask', figsize=(10,4))
            #     plt.imshow(pred.squeeze(0).permute(1, 2, 0).cpu().numpy())
            #     plt.axis("off")
            #     plt.savefig(os.path.join(f"outputs/mask3/mask_{step}.png"))
            #     # plt.show()

            # #########
            auc_score = None

            score = score_func(target_latents, masked_noise_pred_cfg).squeeze(0)


            # score = structural_similarity(target_latents, masked_noise_pred_cfg, multichannel=True,channel_axis=1,data_range=1)

            res += mask * score
        res = res
        if get_auc_score:
            auc_score = self.auc_run(auc_mode, input_latent, target_latents, res, t, hidden_states, random=False,
                                     verbose=0, save_to="outputs/auc")
        else:
            auc_score = None
        return res, auc_score

    @torch.no_grad()
    def auc_run(self, mode, input_latent, target_pred, x_input,x_input_condition, self_cond,explanation, t, encoder_hidden,alpha_res, beta_noise, noise_sal,random=False, verbose=1,
                stride=None, save_to=None, last=True,file_name=None):
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        if stride == None:
            stride = input_latent.shape[-1]
        n_steps = (input_latent.shape[-1] ** 2 + stride - 1) // stride

        saliency_map = F.relu(explanation).unsqueeze(0).unsqueeze(0)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        # normalization
        saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)

        assert mode in ['del', 'ins', 'noise']
        if mode == 'del':
            substrate_fn = torch.zeros_like
            # substrate_fn = torch.rand_like
            # substrate_fn = torch.from_numpy(np.random.uniform(0, 1, size=(input_latent.shape[-1], input_latent.shape[-1])))
        elif mode == 'ins':
            klen = 11
            ksig = 5
            kern = gkern(klen, ksig).cuda()
            substrate_fn = lambda x: torch.nn.functional.conv2d(x, kern, padding=klen // 2).cuda()
        elif mode == 'noise':
            substrate_fn = torch.from_numpy(np.random.uniform(0, 1, size=input_latent.shape))

        if mode == 'del':
            title = 'Deletion game'
            ylabel = 'Pixels deleted'
            start = input_latent.clone()
            finish = substrate_fn(input_latent)
        elif mode == 'ins':
            title = 'Insertion game'
            ylabel = 'Pixels inserted'
            start = substrate_fn(input_latent)
            finish = input_latent.clone()
        elif mode == 'noise':
            title = 'Noising game'
            ylabel = 'Pixels noised'
            start = input_latent.clone()
            finish = substrate_fn

        scores = np.empty(n_steps + 1)

        pfw.set_config(batch_size=1, device=target_pred.device)
        h, w = target_pred.shape[2:]
        target_img = target_pred

        if random:
            saliency_map = torch.rand_like(saliency_map)
        explanation = saliency_map.clone().detach().cpu().numpy()
        salient_order = np.flip(np.argsort(explanation.reshape(-1, n_steps ** 2), axis=1), axis=-1)

        batches = self.num_samples
        for i in range(n_steps + 1):

            input_latents = start.cuda()
            #pred = list(self.ema.ema_model.sample(
            #    input_latents, batch_size=batches, last=last))
            preds = self.model_predictions(
                x_input, input_latents, t, x_input_condition, self_cond)
            pred_res = preds.pred_res
            pred_noise = preds.pred_noise
            x_start = preds.pred_x_start
            pred= input_latents - alpha_res * pred_res - beta_noise * pred_noise + noise_sal

            score = pfw.fid(pred, target_img)
            #print(score)
            # score = cos(pred, target_img)
            scores[i] = score.mean()
            if i == n_steps:
                plt.figure(figsize=(10, 5))
                # plt.title('{} {:.1f}%, P={:.4f}'.format(ylabel, 100 * i / n_steps, scores[i]))
                if verbose == 0:
                    # plt.subplot(339)

                    plt.plot(np.arange(i + 1) / n_steps, scores[:i + 1])
                    # plt.xlim(-0.1, 1.1)
                    # plt.ylim(, 1.05)
                    plt.fill_between(np.arange(i + 1) / n_steps, 0, scores[:i + 1], alpha=0.4)
                    plt.title(title)
                    plt.xlabel(ylabel)
                    plt.ylabel("score")
                    #plt.imshow()
                    if save_to:
                        print(save_to + file_name + mode + '_fid.png')
                        plt.savefig(save_to + file_name + mode + '_fid.png')
                        plt.close()
                    else:
                        plt.show()
                return scores

            coords = salient_order[:, stride * i:stride * (i + 1)]
            #print(start.shape)
            #print(coords.shape)
            """"
            start = start.cpu().numpy().reshape(1, 4, n_steps ** 2)

            start[0, :, coords] = finish.cpu().numpy().reshape(1, 4, n_steps ** 2)[0, :, coords]
            start = torch.from_numpy(start.reshape(1, 4, n_steps, n_steps))
            """
            start = start.cpu().numpy().reshape(1, 3, n_steps ** 2)

            start[0, :, coords] = finish.cpu().numpy().reshape(1, 3, n_steps ** 2)[0, :, coords]
            start = torch.from_numpy(start.reshape(1, 3, n_steps, n_steps))
            # start.cpu().numpy().reshape(1, 4, n_steps**2)[0, :, coords] = finish.cpu().numpy().reshape(1, 4, n_steps**2)[0, :, coords]
            # start.cpu().numpy().reshape(-1, n_steps**2)[:, coords] = finish.cpu().numpy().reshape(-1, n_steps**2)[:, coords]
        return scores

    def gau_masking_latents(self, latents, prob_thresh):
        channel, image_w, image_h = latents.shape[1:]
        mask = (np.random.uniform(0, 1, size=(image_w, image_h)) < prob_thresh).astype(np.float32)
        ltnt = latents.permute(2, 3, 1, 0).squeeze(-1)
        masked_latent = (ltnt.to(torch.float32) * torch.from_numpy(np.dstack([mask] * channel)).to(ltnt.device))
        masked_latent = masked_latent.permute(2, 0, 1).unsqueeze(0).to(torch.float16)
        mask = torch.from_numpy(mask).to(ltnt.device)
        return mask, masked_latent

    def actv_masking_latents(self, latents, prob_thresh, activ):
        image_w, image_h = latents.shape[2:]

        actv_pred_uncond, actv_pred_text = activ.data.chunk(2)
        activations_ = actv_pred_uncond + 0.7 * (actv_pred_text - actv_pred_uncond)

        activation_map = activations_.sum(1).unsqueeze(0)
        activation_map = F.interpolate(activation_map, size=(image_w, image_h), mode='bilinear', align_corners=False)

        mean = activation_map.mean()
        dice = np.random.randint(0, 4)
        if dice == 0:
            actv_mask = torch.where(activation_map < mean, 1, 0).squeeze(0).squeeze(0).cpu().detach().numpy()
            mask = np.random.uniform(0, 1, size=(image_w, image_h))
            mask = mask * actv_mask
            mask = (mask > 0.1).astype(np.float32)

        elif dice == 1:
            actv_mask = torch.where(activation_map > mean, 1, 0).squeeze(0).squeeze(0).cpu().detach().numpy()
            mask = np.random.uniform(0, 1, size=(image_w, image_h))
            mask = mask * actv_mask
            mask = (mask > 0.3).astype(np.float32)

        else:
            mask = (np.random.uniform(0, 1, size=(image_w, image_h)) < prob_thresh).astype(np.float16)

        ltnt = latents.permute(2, 3, 1, 0).squeeze(-1)
        masked_latent = (ltnt.to(torch.float32) * torch.from_numpy(np.dstack([mask] * 4)).to(ltnt.device))
        masked_latent = masked_latent.permute(2, 0, 1).unsqueeze(0).to(torch.float16)
        mask = torch.from_numpy(mask).to(ltnt.device)
        return mask, masked_latent



# trainer class


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        augment_flip=True,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=25,
        results_folder='/mnt/data/result_ge47nej/results_translation_train/pred_res_noise_ssim',
        amp=False,
        fp16=False,
        split_batches=True,
        convert_image_to=None,
        condition=False,
        sub_dir=False,
        equalizeHist=False,
        crop_patch=False,
        generation=False,
        num_unet=2,
        normalization_method = 2
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )
        self.sub_dir = sub_dir
        self.crop_patch = crop_patch

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        #assert has_int_squareroot(
        #    num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.condition = condition
        self.num_unet = num_unet

        if self.condition:
            if len(folder) == 3:
                self.condition_type = 1
                # test_input
                ds = dataset(folder[-1], self.image_size,
                             augment_flip=False, convert_image_to=convert_image_to, condition=0, equalizeHist=equalizeHist, crop_patch=crop_patch, sample=True, generation=generation)
                trian_folder = folder[0:2]

                self.sample_dataset = ds

                self.sample_loader = cycle(self.accelerator.prepare(DataLoader(self.sample_dataset, batch_size=num_samples, shuffle=True,
                                                                               pin_memory=True, num_workers=4)))  # cpu_count()

                ds = dataset(trian_folder, self.image_size, augment_flip=augment_flip,
                             convert_image_to=convert_image_to, condition=1, equalizeHist=equalizeHist, crop_patch=crop_patch, generation=generation)
                self.dl = cycle(self.accelerator.prepare(DataLoader(ds, batch_size=train_batch_size,
                                shuffle=True, pin_memory=True, num_workers=4)))
            elif len(folder) == 4:
                self.condition_type = 2
                # test_gt+test_input
                ds = dataset(folder[2:4], self.image_size,
                             augment_flip=False, convert_image_to=convert_image_to, condition=1, equalizeHist=equalizeHist, crop_patch=crop_patch, sample=True, generation=generation)
                trian_folder = folder[0:2]

                self.sample_dataset = ds
                self.sample_dataloader = DataLoader(self.sample_dataset, batch_size=num_samples, shuffle=True,pin_memory=True, num_workers=4)
                self.sample_loader = cycle(self.accelerator.prepare(self.sample_dataloader))  # cpu_count()

                ds = dataset(trian_folder, self.image_size, augment_flip=augment_flip,
                             convert_image_to=convert_image_to, condition=1, equalizeHist=equalizeHist, crop_patch=crop_patch, generation=generation)
                self.train_dataloader = DataLoader(ds, batch_size=train_batch_size,shuffle=True, pin_memory=True, num_workers=4)
                self.dl = cycle(self.accelerator.prepare(self.train_dataloader))
            elif len(folder) == 6:
                self.condition_type = 3
                # test_gt+test_input
                ds = dataset(folder[3:6], self.image_size,
                             augment_flip=False, convert_image_to=convert_image_to, condition=2, equalizeHist=equalizeHist, crop_patch=crop_patch, sample=True, generation=generation)
                trian_folder = folder[0:3]

                self.sample_dataset = ds
                self.sample_loader = cycle(self.accelerator.prepare(DataLoader(self.sample_dataset, batch_size=num_samples, shuffle=True,
                                                                               pin_memory=True, num_workers=4)))  # cpu_count()

                ds = dataset(trian_folder, self.image_size, augment_flip=augment_flip,
                             convert_image_to=convert_image_to, condition=2, equalizeHist=equalizeHist, crop_patch=crop_patch, generation=generation)
                self.dl = cycle(self.accelerator.prepare(DataLoader(ds, batch_size=train_batch_size,
                                shuffle=True, pin_memory=True, num_workers=4)))
        else:
            self.condition_type = 0
            trian_folder = folder

            ds = dataset(trian_folder, self.image_size, augment_flip=augment_flip,
                         convert_image_to=convert_image_to, condition=0, equalizeHist=equalizeHist, crop_patch=crop_patch, generation=generation)
            self.dl = cycle(self.accelerator.prepare(DataLoader(ds, batch_size=train_batch_size,
                            shuffle=True, pin_memory=True, num_workers=4)))
        self.saample_data_len = len(self.sample_dataloader)
        self.train_data_len = len(self.train_dataloader)
        # optimizer

        # self.opt = Adam(diffusion_model.parameters(),
        #                 lr=train_lr, betas=adam_betas)
        if self.num_unet == 1:
            self.opt0 = RAdam(diffusion_model.parameters(),
                              lr=train_lr, weight_decay=0.0)
        elif self.num_unet == 2:
            self.opt0 = RAdam(
                diffusion_model.model.unet0.parameters(), lr=train_lr, weight_decay=0.0)
            self.opt1 = RAdam(
                diffusion_model.model.unet1.parameters(), lr=train_lr, weight_decay=0.0)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay,
                           update_every=ema_update_every)

            self.set_results_folder(results_folder)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        if self.num_unet == 1:
            self.model, self.opt = self.accelerator.prepare(
                self.model, self.opt0)
        elif self.num_unet == 2:
            self.model, self.opt0, self.opt1 = self.accelerator.prepare(
                self.model, self.opt0, self.opt1)
        device = self.accelerator.device
        self.device = device

        if normalization_method == 0:
            # Reinhard
            method = 'Reinhard'
            self.normalizer = stainNorm_Reinhard.Normalizer()
        elif normalization_method == 1:
            # Macenko
            method = 'Macenko'
            self.normalizer = stainNorm_Macenko.Normalizer()
        elif normalization_method == 2:
            # Vahadane
            method = 'Vahadane'
            self.normalizer = stainNorm_Vahadane.Normalizer()
        else:
            print('enter valid normalization method (Reinhard [0], Macenko [1], Vahadane [2])')
            exit()

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
        if self.num_unet == 1:
            data = {
                'step': self.step,
                'model': self.accelerator.get_state_dict(self.model),
                'opt0': self.opt0.state_dict(),
                'ema': self.ema.state_dict(),
                'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
            }
        elif self.num_unet == 2:
            data = {
                'step': self.step,
                'model': self.accelerator.get_state_dict(self.model),
                'opt0': self.opt0.state_dict(),
                'opt1': self.opt1.state_dict(),
                'ema': self.ema.state_dict(),
                'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
            }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        #path = Path(self.results_folder / f'model-{milestone}.pt')
        accelerator = self.accelerator
        if accelerator.is_main_process:
            path = Path(milestone)

            if path.exists():
                data = torch.load(
                    str(path), map_location=self.device)

                model = self.accelerator.unwrap_model(self.model)
                model.load_state_dict(data["model"])
                #model = model.half()



                self.step = data['step']
                if self.num_unet == 1:
                    self.opt0.load_state_dict(data['opt0'])
                elif self.num_unet == 2:
                    self.opt0.load_state_dict(data['opt0'])
                    self.opt1.load_state_dict(data['opt1'])
                self.ema.load_state_dict(data['ema'])

                if exists(self.accelerator.scaler) and exists(data['scaler']):
                    self.accelerator.scaler.load_state_dict(data['scaler'])

                print("load model - "+str(path))

        # self.ema.to(self.device)

    def train(self,log_obj=None):
        accelerator = self.accelerator
        iterations = 0
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step <= self.train_num_steps:

                if self.num_unet == 1:
                    total_loss = [0]
                elif self.num_unet == 2:
                    total_loss = [0, 0]
                for _ in range(self.gradient_accumulate_every):
                    if self.condition:
                        data = next(self.dl)
                        data = [item.to(self.device) for item in data]
                        """
                        if accelerator.is_main_process:
                            all_images = torch.cat(data)
                            save_path = '/mnt/data/result_ge47nej/result_XAI_test/heat_noise'
                            file_name = f'dataset_test_{self.step}.png'
                            utils.save_image(all_images,
                                save_path +'/'+ file_name, nrow=4)
                            print(data)
                        """

                    else:
                        data = next(self.dl)
                        data = data[0] if isinstance(data, list) else data
                        data = data.to(self.device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        for i in range(self.num_unet):
                            loss[i] = loss[i] / self.gradient_accumulate_every
                            total_loss[i] = total_loss[i] + loss[i].item()

                    for i in range(self.num_unet):
                        self.accelerator.backward(loss[i])

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                accelerator.wait_for_everyone()

                if self.num_unet == 1:
                    self.opt0.step()
                    self.opt0.zero_grad()
                elif self.num_unet == 2:
                    self.opt0.step()
                    self.opt0.zero_grad()
                    self.opt1.step()
                    self.opt1.zero_grad()

                accelerator.wait_for_everyone()



                if accelerator.is_main_process:
                    self.ema.to(self.device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        milestone = self.step // self.save_and_sample_every
                        #self.sample(milestone)

                        if self.step != 0 and self.step % (self.save_and_sample_every) == 0:
                            self.save(milestone)
                            # results_folder = self.results_folder
                            # gen_img = './results/test_timestep_10_' + \
                            #     str(milestone)+"_pt"
                            # self.set_results_folder(gen_img)
                            # self.test(last=True, FID=True)
                            # os.system(
                            #     "python fid_and_inception_score.py "+gen_img)
                            # self.set_results_folder(results_folder)
                    if self.num_unet == 1:
                        pbar.set_description(f'loss_unet0: {total_loss[0]:.4f}')
                    elif self.num_unet == 2:
                        pbar.set_description(
                            f'loss_unet0: {total_loss[0]:.4f},loss_unet1: {total_loss[1]:.4f}')
                    #log_obj.log({"loss_unet0": total_loss[0], "loss_unet1": total_loss[1]})
                    iterations += self.gradient_accumulate_every
                    if iterations >= self.train_data_len:
                        #print(iterations, self.train_data_len)
                        self.step += 1
                        iterations = 0
                        pbar.update(1)

        accelerator.print('training complete')

    def sample(self, milestone, last=True, FID=False):
        self.ema.ema_model.eval()
        with torch.no_grad():
            batches = self.num_samples
            if self.condition_type == 0:
                x_input_sample = [0]
                show_x_input_sample = []
            elif self.condition_type == 1:
                x_input_sample = [next(self.sample_loader).to(self.device)]
                show_x_input_sample = x_input_sample
            elif self.condition_type == 2:
                x_input_sample = next(self.sample_loader)
                x_input_sample = [item.to(self.device)
                                  for item in x_input_sample]
                show_x_input_sample = x_input_sample
                x_input_sample = x_input_sample[1:]
            elif self.condition_type == 3:
                x_input_sample = next(self.sample_loader)
                x_input_sample = [item.to(self.device)
                                  for item in x_input_sample]
                show_x_input_sample = x_input_sample
                x_input_sample = x_input_sample[1:]

            all_images_list = show_x_input_sample + \
                list(self.ema.ema_model.sample(
                    x_input_sample, batch_size=batches, last=last))

            all_images = torch.cat(all_images_list, dim=0)

            if last:
                nrow = int(math.sqrt(self.num_samples))
            else:
                nrow = all_images.shape[0]

            if FID:
                for i in range(batches):
                    file_name = f'sample-{milestone}.png'
                    utils.save_image(
                        all_images_list[0][i].unsqueeze(0), os.path.join(self.results_folder, file_name), nrow=1)
                    milestone += 1
                    if milestone >= self.total_n_samples:
                        break
            else:
                file_name = f'sample-{milestone}.png'
                utils.save_image(all_images, str(
                    self.results_folder / file_name), nrow=nrow)
            print("sampe-save "+file_name)
        self.ema.ema_model.train()
        return milestone

    def test(self, save_heatmap_path,save_result_folder_sample,sample=False, last=True, FID=False,XAI= False):
        self.ema.ema_model.init()
        self.ema.to(self.device)
        print("test start")
        result_folder_heat_noise = save_heatmap_path
        result_folder_sample = save_result_folder_sample
        self.set_results_folder2(result_folder_heat_noise)
        self.set_results_folder2(result_folder_sample)
        psnr_list = []
        ssim_list = []
        predicted_list = []
        transform = T.Compose([
            transforms.Resize([224, 224]),  # 将图片统一尺寸
            # transforms.RandomHorizontalFlip(),
            # 将图片随机水平翻转，推理时无需增强，保存时用acc做依据，当数据不平衡时用f1 score或roc，补充一个垂直翻转增强，vertical，
            # 控制图像被数据增强的概率,选择p=0.3，保留最好的model，用resnet评估RDDM生成结果
            # transforms.ToTensor(),  # 将图片转换为tensor
            transforms.Normalize(  # 标准化处理—>转换为正态分布，使模型更容易收敛，不需要重新计算
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        resnet_model = ResNet50(block=ResNetblock, num_classes=2).to(self.device)
        weight_path = '/mnt/data/result_ge47nej/resnet/ckpt_full_model/model.pth'
        resnet_model = load(resnet_model, weight_path)
        if self.condition:
            print(self.condition)
            self.ema.ema_model.eval()
            loader = DataLoader(
                dataset=self.sample_dataset,
                batch_size=1)
            size = len(loader)
            i = 0
            acc = 0
            target0 = 0
            target1 = 0
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            auc = 0.


            for items in loader:
                #print(items)
                if self.condition:
                    file_name = self.sample_dataset.load_name(
                        i, sub_dir=self.sub_dir)
                    file_name = f'{i}.png' if file_name==None else file_name
                else:
                    file_name = f'{i}.png'
                i += 1

                with torch.no_grad():
                    batches = self.num_samples

                    if self.condition_type == 0:
                        x_input_sample = [0]
                        show_x_input_sample = []
                    elif self.condition_type == 1:
                        x_input_sample = [items.to(self.device)]
                        show_x_input_sample = x_input_sample
                    elif self.condition_type == 2:
                        x_input_sample = [item.to(self.device)
                                          for item in items]
                        show_x_input_sample = x_input_sample
                        x_input_sample = x_input_sample[1:]
                    elif self.condition_type == 3:
                        x_input_sample = [item.to(self.device)
                                          for item in items]
                        show_x_input_sample = x_input_sample
                        x_input_sample = x_input_sample[1:]

                    if sample:
                        all_images_list= show_x_input_sample + \
                            list(self.ema.ema_model.sample(
                                x_input_sample, batch_size=batches))
                    else:
                        all_images_list_base,heatmap_list,auc_list, heatmap_final = list(self.ema.ema_model.sample(
                            x_input_sample, batch_size=batches, last=last,file_name = file_name,xai=XAI))
                        #print(auc_list)
                        all_images_list = all_images_list_base
                        all_images_list2 = show_x_input_sample + all_images_list_base
                        if self.crop_patch:
                            k = 0
                            for img in all_images_list:
                                pad_size = self.sample_dataset.get_pad_size(i)
                                _, _, h, w = img.shape
                                img = img[:, :, 0:h -
                                          pad_size[0], 0:w-pad_size[1]]
                                all_images_list[k] = img
                                k += 1

                    all_images = torch.cat(all_images_list2, dim=0)

                    processed_img = transform(all_images_list2[-1])
                    predicted_label = resnet_model(processed_img)
                    predicted_list = predicted_label
                    #print(file_name)
                    #print(predicted_label)
                    target_label = file_name.split('\\')[-1].split('.')[0].split('_')[-1][0]
                    if target_label =='0' or target_label == '1':
                        target_label = 0
                        target0 += 1
                    else:
                        target_label = 1
                        target1 += 1

                    acc += (predicted_label.argmax(1) == target_label).sum().item()

                    if predicted_label.argmax(1) == target_label == 1:

                        TP += 1
                    elif predicted_label.argmax(1) == target_label == 0:
                        TN += 1
                    elif predicted_label.argmax(1) == 1 and target_label == 0:
                        FP += 1
                    else:
                        FN += 1

                    normalized_img = all_images_list2[-1]
                    #print(all_images_list2[0],normalized_img)
                    psnr, ssim = self.evaluate(all_images_list2[0], normalized_img)
                    psnr_list.append(psnr)
                    ssim_list.append(ssim)
                    print("psnr:{},ssim:{}".format(psnr, ssim))




                    if last:
                        nrow = int(math.sqrt(self.num_samples))
                    else:
                        nrow = all_images.shape[0]

                    #utils.save_image(all_images, result_folder_sample + str(file_name), nrow=nrow)
                    print(result_folder_sample + str(file_name))
                    #print(all_heatmap.shape)
                    if XAI:
                        #auc += auc_list
                        #print("auc_list:{}".format(auc_list))
                        all_heatmap = torch.cat(heatmap_list, dim=0)
                        #all_heatmap = heatmap_list[-1]
                        utils.save_image(all_heatmap,result_folder_heat_noise+"/"+str(file_name),nrow=15)
                        #print(resize_images.shape)
                        resize_images = all_images_list_base[-1]
                        #utils.save_image(resize_images,result_folder_heat_noise+"/"+'resize_'+str(file_name),nrow = nrow)
                        #utils.save_image(heatmap_final, result_folder_heat_noise + "/" + 'resize_heatmap' + str(file_name), nrow=nrow)
                        #print(all_images_list[-1].squeeze().permute(1, 2, 0).shape)
                        cam = self.show_mask_on_image(img=all_images_list[-1].squeeze().permute(1, 2, 0).cpu(),mask=heatmap_list[-1].squeeze().permute(1, 2, 0).cpu())
                        batchsize, c, h, w = heatmap_list[0].shape

                        # print(heat_res[0].shape)
                        heat_map_len = len(heatmap_list)
                        plt.figure(result_folder_heat_noise + '/' + str(file_name), figsize=(h / 100, w / 100),
                                   dpi=100)
                        """
                        for i in range(heat_map_len):
                            plt.subplot(1, heat_map_len, i + 1)
                            # print(heat_noise[i].shape)
                            img = heat_map[i]
                            # print(img.shape)
                            img = img.squeeze().permute(1, 2, 0)
                            # print(img.shape)
                            plt.imshow(img)
                            plt.axis("off")
                        """
                        """
                        img = heatmap_list[-1]
                        img = img.squeeze().permute(1, 2, 0)
        
                        plt.imshow(all_images_list[-1].squeeze().permute(1, 2, 0).cpu())
                        plt.imshow(img.cpu(), alpha=0.2, cmap='coolwarm')
                        """
                        #plt.imshow(cam)

                        #plt.savefig(result_folder_heat_noise + '/' + 'plt_cam_' + str(file_name))
                        print(result_folder_heat_noise + '/' +'plt_cam_'+ str(file_name))
                    print("test-save "+file_name + result_folder_sample)
        else:
            if FID:
                self.total_n_samples = 50000
                img_id = len(glob.glob(f"{self.results_folder}/*"))
                n_rounds = (self.total_n_samples -
                                    img_id) // self.num_samples+1
            else:
                n_rounds = 100
            for i in range(n_rounds):
                if FID:
                        i = img_id
                img_id = self.sample(i, last=last, FID=FID)


        psnr_mean = np.mean(psnr_list)
        psnr_std = np.std(psnr_list)
        ssim_mean = np.mean(ssim_list)
        ssim_std = np.std(ssim_list)
        print("psnr_mean:{},psnr_std:{}".format(psnr_mean, psnr_std))
        print("ssim_mean:{},ssim_std_{}".format(ssim_mean, ssim_std))
        #print("auc_mean:{}".format(auc_mean))

       # print("avg_deg:{},SFS:{}".format(avg_deg,SFS))
        print("test end")
    # for overlap heatmap and img
    def show_mask_on_image(self,img, mask):
        cam = np.uint8(mask) * 0.5 + np.float32(img)

        cam = cam / np.max(cam)
        return np.uint8(255 * cam)

    def set_results_folder(self, path):
        self.results_folder = Path(path)
        if not self.results_folder.exists():
            os.makedirs(self.results_folder)

    def set_results_folder2(self, path):
        results_folder = Path(path)
        if not results_folder.exists():
            os.makedirs(results_folder)


    def evaluate(self,img1,img2):

        img1 = img1.squeeze().permute(1,2,0).detach().cpu().numpy()
        img2 = img2.squeeze().permute(1,2,0).detach().cpu().numpy()
        print(img1.shape,img2.shape)
        psnr = peak_signal_noise_ratio(img1,img2)
        ssim = structural_similarity(img1, img2, multichannel=True,channel_axis=-1,data_range=1)
        return psnr,ssim
