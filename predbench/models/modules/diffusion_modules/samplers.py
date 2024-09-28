# SMLD: s = -1/sigma * z
# DDPM: s = -1/sqrt(1 - alpha) * z
# All `scorenet` models return z, not s!

from typing import Any
import torch
import logging
import numpy as np

from functools import partial
from scipy.stats import hmean
from torch.distributions.gamma import Gamma
from tqdm import tqdm
from . import pndm
import torch.nn as nn
from predbench.registry import MODELS



def get_sigmas(num_classes, sigma_dist, sigma_begin, sigma_end, device):

    if sigma_dist == 'geometric':
        return torch.logspace(np.log10(sigma_begin), np.log10(sigma_end),num_classes).to(device)

    elif sigma_dist == 'linear':
        return torch.linspace(sigma_begin, sigma_end,num_classes).to(device)

    elif sigma_dist == 'cosine':
        t = torch.linspace(num_classes, 0, num_classes+1)/num_classes
        s = 0.008
        f = torch.cos((t + s)/(1 + s) * np.pi/2)**2
        return f[:-1]/f[-1]

    else:
        raise NotImplementedError('sigma distribution not supported')


@torch.no_grad()
def fpndm_sampler(
    x_mod, score_net, cond=None, final_only=False, subsample_steps=None, clip_before=True
):
    alphas, alphas_prev, betas = score_net.alphas, score_net.alphas_prev, score_net.betas
    steps = np.arange(len(betas))

    alphas_old = alphas.flip(0)
    
    # New ts (see https://github.com/ermongroup/ddim/blob/main/runners/diffusion.py)
    skip = len(alphas) // subsample_steps
    steps = range(0, len(alphas), skip)
    steps_next = [-1] + list(steps[:-1])

    
    steps = torch.tensor(steps, device=alphas.device)
    steps_next = torch.tensor(steps_next, device=alphas.device)
    alphas_next = alphas.index_select(0, steps_next + 1)
    alphas = alphas.index_select(0, steps + 1)
    

    images = []
    score_net = partial(score_net, cond=cond)

    ets = []
    for i, step in enumerate(steps):

        t_ = (steps[i] * torch.ones(x_mod.shape[0], device=x_mod.device)).long()
        t_next = (steps_next[i] * torch.ones(x_mod.shape[0], device=x_mod.device)).long()
        x_mod, ets = pndm.gen_order_4(
            x_mod, t_, t_next, model=score_net, alphas_cump=alphas_old, ets=ets, clip_before=clip_before
        )

        if not final_only:
            images.append(x_mod.to('cpu'))
            images.append(x_mod)

    if final_only:
        return x_mod.unsqueeze(0)
    else:
        return torch.stack(images)


@torch.no_grad()
def ddim_sampler(
    x_mod, score_net, cond=None, final_only=False, subsample_steps=None, clip_before=True, 
    denoise=True, t_min=-1, gamma=False, 
):
    alphas, alphas_prev, betas = score_net.alphas, score_net.alphas_prev, score_net.betas
    if gamma:
        ks_cum, thetas = score_net.k_cum, score_net.theta_t
    steps = np.arange(len(betas))

    # New ts (see https://github.com/ermongroup/ddim/blob/main/runners/diffusion.py) 
    if subsample_steps is not None:
        if subsample_steps < len(alphas):
            skip = len(alphas) // subsample_steps
            steps = range(0, len(alphas), skip)
            steps = torch.tensor(steps, device=alphas.device)
            # new alpha, beta, alpha_prev
            alphas = alphas.index_select(0, steps)
            alphas_prev = torch.cat([alphas[1:], torch.tensor([1.0]).to(alphas)])
            betas = 1.0 - torch.div(alphas, alphas_prev) # for some reason we lose a bit of precision here
            if gamma:
                ks_cum = ks_cum.index_select(0, steps)
                thetas = thetas.index_select(0, steps)

    images = []
    score_net = partial(score_net, cond=cond)
    x_transf = False

    L = len(steps)
    for i, step in enumerate(steps):

        if step < t_min*len(alphas): # otherwise, wait until it happens
            continue

        if not x_transf and t_min > 0: # we must add noise to the previous frame
            if gamma:
                z = Gamma(torch.full(x_mod.shape[1:], ks_cum[i]),
                          torch.full(x_mod.shape[1:], 1 / thetas[i])).sample((x_mod.shape[0],)).to(x_mod.device)
                z = (z - ks_cum[i]*thetas[i])/((1 - alphas[i]).sqrt())
            else:
                z = torch.randn_like(x_mod)
            x_mod = alphas[i].sqrt() * x_mod + (1 - alphas[i]).sqrt() * z
        x_transf = True

        c_beta, c_alpha, c_alpha_prev = betas[i], alphas[i], alphas_prev[i]
        labels = (step * torch.ones(x_mod.shape[0], device=x_mod.device)).long()
        epsilon = score_net(x_mod, labels)

        x0 = (1 / c_alpha.sqrt()) * (x_mod - (1 - c_alpha).sqrt() * epsilon)
        if clip_before:
            x0 = x0.clip_(-1, 1)
        x_mod = c_alpha_prev.sqrt() * x0 + (1 - c_alpha_prev).sqrt() * epsilon

        if not final_only:
            # images.append(x_mod.to('cpu'))
            images.append(x_mod)

    if denoise: # x + batch_mul(std ** 2, score_fn(x, eps_t))
        last_noise = ((L - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)).long()
        x_mod = x_mod - (1 - alphas[-1]).sqrt() * score_net(x_mod, last_noise)
        if not final_only:
            # images.append(x_mod.to('cpu'))
            images.append(x_mod)

    if final_only:
        return x_mod.unsqueeze(0)
    else:
        return torch.stack(images)

@torch.no_grad()
def ddpm_sampler(
    x_mod, score_net, cond=None, final_only=False, subsample_steps=None, clip_before=True, 
    denoise=True, t_min=-1, gamma=False, frac_steps=None, just_beta=False, 
):
    alphas, alphas_prev, betas = score_net.alphas, score_net.alphas_prev, score_net.betas
    steps = np.arange(len(betas))
    if gamma:
        ks_cum, thetas = score_net.k_cum, score_net.theta_t


    # New ts (see https://github.com/ermongroup/ddim/blob/main/runners/diffusion.py)
    if subsample_steps is not None:
        if subsample_steps < len(alphas):
            skip = len(alphas) // subsample_steps
            steps = range(0, len(alphas), skip)
            steps = torch.tensor(steps, device=alphas.device)
            # new alpha, beta, alpha_prev
            alphas = alphas.index_select(0, steps)
            alphas_prev = torch.cat([alphas[1:], torch.tensor([1.0]).to(alphas)])
            betas = 1.0 - torch.div(alphas, alphas_prev) # for some reason we lose a bit of precision here
            if gamma:
                ks_cum = ks_cum.index_select(0, steps)
                thetas = thetas.index_select(0, steps)

    # Subsample steps : keep range but decrease number of steps
    #if subsample_steps is not None:
    #    steps = torch.round(torch.linspace(0, len(alphas)-1, subsample_steps)).long()
    #    alphas = alphas[steps]
    #    alphas_prev = torch.cat([alphas[1:], torch.tensor([1.0]).to(alphas)])
    #    betas = 1 - alphas/alphas_prev

    # Frac steps : fraction of last steps to cover
    if frac_steps is not None:
        steps = steps[int((1 - frac_steps)*len(steps)):]
        alphas = alphas[steps]
        alphas_prev = alphas_prev[steps]
        betas = betas[steps]
        if gamma:
            ks_cum = ks_cum[steps]
            thetas = thetas[steps]

    images = []
    score_net = partial(score_net, cond=cond)
    x_transf = False

    L = len(steps)
    for i, step in enumerate(steps):

        if step < t_min * len(alphas): # wait until it happens
            continue

        if not x_transf and t_min > 0: # we must add noise to the previous frame
            if gamma:
                z = Gamma(torch.full(x_mod.shape[1:], ks_cum[i]),
                          torch.full(x_mod.shape[1:], 1 / thetas[i])).sample((x_mod.shape[0],)).to(x_mod.device)
                z = (z - ks_cum[i]*thetas[i]) / (1 - alphas[i]).sqrt()
            else:
                z = torch.randn_like(x_mod)
            x_mod = alphas[i].sqrt() * x_mod + (1 - alphas[i]).sqrt() * z
        x_transf = True

        c_beta, c_alpha, c_alpha_prev = betas[i], alphas[i], alphas_prev[i]
        labels = (step * torch.ones(x_mod.shape[0], device=x_mod.device)).long()
        epsilon = score_net(x_mod, labels)

        # x_mod = 1 / (1 - c_beta).sqrt() * (x_mod + c_beta / (1 - c_alpha).sqrt() * grad)
        x0 = (1 / c_alpha.sqrt()) * (x_mod - (1 - c_alpha).sqrt() * epsilon)
        if clip_before:
            x0 = x0.clip_(-1, 1)
        x_mod = (c_alpha_prev.sqrt() * c_beta / (1 - c_alpha)) * x0 + ((1 - c_beta).sqrt() * (1 - c_alpha_prev) / (1 - c_alpha)) * x_mod

        if not final_only:
            # images.append(x_mod.to('cpu'))
            images.append(x_mod)

        # If last step, don't add noise
        last_step = (i + 1 == L)
        if last_step:
            continue

        # Add noise
        if gamma:
            z = Gamma(torch.full(x_mod.shape[1:], ks_cum[i]),
                        torch.full(x_mod.shape[1:], 1 / thetas[i])).sample((x_mod.shape[0],)).to(x_mod.device)
            noise = (z - ks_cum[i]*thetas[i])/((1 - alphas[i]).sqrt())
        else:
            noise = torch.randn_like(x_mod)
        if just_beta:
            x_mod += c_beta.sqrt() * noise
        else:
            x_mod += ((1 - c_alpha_prev) / (1 - c_alpha) * c_beta).sqrt() * noise

    # Denoise 
    if denoise: 
        last_noise = ((L - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)).long()
        x_mod = x_mod - (1 - alphas[-1]).sqrt() * score_net(x_mod, last_noise)
        if not final_only:
            # images.append(x_mod.to('cpu'))
            images.append(x_mod)

    if final_only:
        return x_mod.unsqueeze(0)
    else:
        return torch.stack(images)

@torch.no_grad()
def anneal_Langevin_dynamics(
    x_mod, score_net, cond=None, final_only=False, denoise=True, n_steps_each=200, step_lr=0.000008, 
    harm_mean=False, same_noise=False, noise_val=None, frac_steps=None,
):
    sigmas = score_net.sigmas
    steps = np.arange(len(sigmas))

    if harm_mean:
        sigmas_hmean = hmean(sigmas.cpu())

    # Sub steps : fraction of last steps to cover
    if frac_steps is not None:
        steps = steps[int((1 - frac_steps)*len(steps)):]
        sigmas = sigmas[steps]

    if same_noise and noise_val is None:
        noise_val = x_mod.detach().clone()

    images = []

    score_net = partial(score_net, cond=cond)

    for c, sigma in enumerate(sigmas):
        labels = (torch.ones(x_mod.shape[0], device=x_mod.device) * c).long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for s in range(n_steps_each):

            grad = score_net(x_mod, labels)
            if harm_mean:
                grad = grad * sigmas_hmean / sigma

            if same_noise:
                noise = noise_val
            else:
                z = torch.randn_like(x_mod)
                noise = z
            x_mod = x_mod - step_size / sigma * grad + (step_size * 2.).sqrt() * noise

            if not final_only:
                # images.append(x_mod.to('cpu'))
                images.append(x_mod)

    if denoise:
        last_noise = ((len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)).long()
        x_mod = x_mod - sigmas[-1] * score_net(x_mod, last_noise)
        if not final_only:
            # images.append(x_mod.to('cpu'))
            images.append(x_mod)

    if final_only:
        return x_mod.unsqueeze(0)
    else:
        return torch.stack(images)

@torch.no_grad()
def sparse_anneal_Langevin_dynamics(
    x_mod_sparse, sparsity, score_net, cond=None, final_only=False, denoise=True, n_steps_each=200, step_lr=0.000008,
    harm_mean=False, same_noise=False, noise_val=None, frac_steps=None,
):
    sigmas = score_net.sigmas
    steps = np.arange(len(sigmas))

    L = len(sigmas)
    if harm_mean:
        sigmas_hmean = hmean(sigmas.cpu())

    # Sub steps : fraction of last steps to cover
    if frac_steps is not None:
        steps = steps[int((1 - frac_steps)*len(steps)):]
        sigmas = sigmas[steps]

    if same_noise and noise_val is None:
        noise_val = x_mod.detach().clone()

    images = []
    x_mod = x_mod_sparse.clone()

    score_net = partial(score_net, cond=cond)

    for c, sigma in enumerate(sigmas):
        labels = (torch.ones(x_mod.shape[0], device=x_mod.device) * c).long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for s in range(n_steps_each):

            grad = score_net(x_mod, labels)
            if harm_mean:
                grad = grad * sigmas_hmean / sigma

            if same_noise:
                noise = noise_val
            else:
                z = torch.randn_like(x_mod)
                noise = z

            x_mod = x_mod - step_size / sigma * grad + (step_size * 2.).sqrt() * noise
            x_mod_sparse = x_mod_sparse - step_size / sigma * (1/sparsity * grad) + (step_size * 2.).sqrt() * (sparsity * noise)

            # grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            # image_norm = torch.norm(x_mod.reshape(x_mod.shape[0], -1), dim=-1).mean()
            # noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            # snr = (step_size / 2.).sqrt() * grad_norm / noise_norm
            # grad_mean_norm = torch.norm(grad.mean(dim=0).reshape(-1)) ** 2 * sigma ** 2

            if not final_only:
                # images.append(x_mod_sparse.to('cpu'))
                images.append(x_mod_sparse)

            # if (c == 0 and s == 0) or (c*n_steps_each+s+1) % max((L*n_steps_each)//10, 1) == 0:
            #     if verbose:
            #         print("ALS level: {:.04f}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
            #             (c*n_steps_each+s+1)/(L*n_steps_each), step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))
            #     if log:
            #         logging.info("ALS level: {:.04f}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
            #             (c*n_steps_each+s+1)/(L*n_steps_each), step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

    if denoise:
        last_noise = ((len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)).long()
        x_mod_sparse = x_mod_sparse - sigmas[-1] * sparsity * score_net(x_mod, last_noise)
        if not final_only:
            # images.append(x_mod_sparse.to('cpu'))
            images.append(x_mod_sparse)

    if final_only:
        return x_mod_sparse.unsqueeze(0)
    else:
        return torch.stack(images)

@torch.no_grad()
def anneal_Langevin_dynamics_consistent(
    x_mod, score_net, cond=None, final_only=False, denoise=True, n_steps_each=200, step_lr=0.000008, 
    harm_mean=False, same_noise=False, noise_val=None, frac_steps=None,
):
    sigmas = score_net.sigmas
    steps = np.arange(len(sigmas))

    L = len(sigmas)
    sigma_begin = sigmas[0].cpu().item()
    sigma_end = sigmas[-1].cpu().item()
    consistent_sigmas = np.geomspace(sigma_begin, sigma_end, (L - 1) * n_steps_each + 1)

    smallest_invgamma = consistent_sigmas[-1] / consistent_sigmas[-2]
    lowerbound = sigmas[-1] ** 2 * (1 - smallest_invgamma)
    higherbound = sigmas[-1] ** 2 * (1 + smallest_invgamma)
    assert lowerbound < step_lr < higherbound, f"Could not satisfy {lowerbound} < {step_lr} < {higherbound}"

    eta = step_lr / (sigmas[-1] ** 2)

    if harm_mean:
        sigmas_hmean = hmean(consistent_sigmas)

    # Sub steps : fraction of last steps to cover
    if frac_steps is not None:
        steps = steps[int((1 - frac_steps)*len(steps)):]
        consistent_sigmas = consistent_sigmas[steps]

    consistent_L = len(consistent_sigmas)
    iter_consistent_sigmas = iter(consistent_sigmas)
    next_sigma = next(iter_consistent_sigmas)

    if same_noise and noise_val is None:
        noise_val = x_mod.detach().clone()

    images = []

    score_net = partial(score_net, cond=cond)

    for c in range(consistent_L):

        c_sigma = next_sigma
        used_sigmas = torch.tensor([c_sigma]*len(x_mod)).reshape(len(x_mod), *([1] * len(x_mod.shape[1:]))).float().to(x_mod.device)
        grad = score_net(x_mod, used_sigmas, y_is_label=False)

        if harm_mean:
            grad = grad * sigmas_hmean / used_sigmas

        x_mod -= eta * c_sigma * grad
        if not final_only:
            # images.append(x_mod.to('cpu'))
            images.append(x_mod)

        last_step = c + 1 == consistent_L
        if last_step:

            if denoise:
                last_noise = ((len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)).long()
                x_mod = x_mod - sigmas[-1] * score_net(x_mod, last_noise)
                if not final_only:
                    # images.append(x_mod.to('cpu'))
                    images.append(x_mod)

                continue

        next_sigma = next(iter_consistent_sigmas)
        gamma = c_sigma/next_sigma
        beta = (1 - (gamma*(1 - eta))**2).sqrt()
        if same_noise:
            noise = noise_val
        else:
            z = torch.randn_like(x_mod)
            noise = z
        x_mod += beta * next_sigma * noise

    if final_only:
        return x_mod.unsqueeze(0)
    else:
        return torch.stack(images)

@torch.no_grad()
def sparse_anneal_Langevin_dynamics_consistent(
    x_mod_sparse, sparsity, score_net, cond=None, final_only=False, denoise=True, n_steps_each=200, step_lr=0.000008,
    harm_mean=False,same_noise=False, noise_val=None, frac_steps=None, 
):
    sigmas = score_net.sigmas
    steps = np.arange(len(sigmas))

    L = len(sigmas)
    sigma_begin = sigmas[0].cpu().item()
    sigma_end = sigmas[-1].cpu().item()
    consistent_sigmas = np.geomspace(sigma_begin, sigma_end, (L - 1) * n_steps_each + 1)

    smallest_invgamma = consistent_sigmas[-1] / consistent_sigmas[-2]
    lowerbound = sigmas[-1] ** 2 * (1 - smallest_invgamma)
    higherbound = sigmas[-1] ** 2 * (1 + smallest_invgamma)
    assert lowerbound < step_lr < higherbound, f"Could not satisfy {lowerbound} < {step_lr} < {higherbound}"

    eta = step_lr / (sigmas[-1] ** 2)

    if harm_mean:
        sigmas_hmean = hmean(consistent_sigmas)

    # Sub steps : fraction of last steps to cover
    if frac_steps is not None:
        steps = steps[int((1 - frac_steps)*len(steps)):]
        consistent_sigmas = consistent_sigmas[steps]

    consistent_L = len(consistent_sigmas)
    iter_consistent_sigmas = iter(consistent_sigmas)
    next_sigma = next(iter_consistent_sigmas)

    if same_noise and noise_val is None:
        noise_val = x_mod.detach().clone()

    images = []
    x_mod = x_mod_sparse.clone()

    score_net = partial(score_net, cond=cond)

    for c in range(consistent_L):

        c_sigma = next_sigma
        used_sigmas = torch.tensor([c_sigma]*len(x_mod)).reshape(len(x_mod), *([1] * len(x_mod.shape[1:]))).float().to(x_mod.device)
        grad = score_net(x_mod, used_sigmas, y_is_label=False)

        if harm_mean:
            grad = grad * sigmas_hmean / used_sigmas

        x_mod += eta * c_sigma**2 * grad
        if not final_only:
            # images.append(x_mod.to('cpu'))
            images.append(x_mod)

        last_step = c + 1 == consistent_L
        if last_step:

            if denoise:
                last_noise = ((len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)).long()
                x_mod = x_mod + sigmas[-1] * score_net(x_mod, last_noise)
                x_mod_sparse = x_mod_sparse + sigmas[-1] * 1/sparsity * score_net(x_mod, last_noise)
                if not final_only:
                    # images.append(x_mod.to('cpu'))
                    images.append(x_mod)

                continue

        next_sigma = next(iter_consistent_sigmas)
        gamma = c_sigma/next_sigma
        beta = (1 - (gamma*(1 - eta))**2).sqrt()
        if same_noise:
            noise = noise_val
        else:
            z = torch.randn_like(x_mod)
            noise = z
        x_mod += next_sigma * beta * noise
        x_mod_sparse += next_sigma * beta * sparsity * noise

    if final_only:
        return x_mod_sparse.unsqueeze(0)
    else:
        return torch.stack(images)



# TODO: Implementation of score-based generation sampler

@MODELS.register_module()
class FPNDMSampler:
    def __init__(self,
        final_only=False, subsample_steps=None, clip_before=True, 
    ):
        self.sampler = partial(ddpm_sampler, 
            final_only=final_only, subsample_steps=subsample_steps, clip_before=clip_before, 
        )
    
    def __call__(self, x_mod, score_net, cond):
        return self.sampler(x_mod=x_mod, score_net=score_net, cond=cond)


@MODELS.register_module()
class DDIMSampler:
    def __init__(self,
        final_only=False, subsample_steps=None, clip_before=True, 
        denoise=True, t_min=-1, gamma=False, 
    ):
        self.sampler = partial(ddpm_sampler, 
            final_only=final_only, subsample_steps=subsample_steps, clip_before=clip_before, 
            denoise=denoise, t_min=t_min, gamma=gamma,  
        )
    
    def __call__(self, x_mod, score_net, cond):
        return self.sampler(x_mod=x_mod, score_net=score_net, cond=cond)


@MODELS.register_module()
class DDPMSampler:
    def __init__(self,
        final_only=False, subsample_steps=None, clip_before=True, 
        denoise=True, t_min=-1, gamma=False, frac_steps=None, just_beta=False, 
    ):
        self.sampler = partial(ddpm_sampler, 
            final_only=final_only, subsample_steps=subsample_steps, clip_before=clip_before, 
            denoise=denoise, t_min=t_min, gamma=gamma, frac_steps=frac_steps, just_beta=just_beta, 
        )
    
    def __call__(self, x_mod, score_net, cond):
        return self.sampler(x_mod=x_mod, score_net=score_net, cond=cond)

