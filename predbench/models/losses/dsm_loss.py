import torch.nn as nn
from typing import List
import torch
from functools import partial
from torch.distributions.gamma import Gamma


from predbench.registry import MODELS



def anneal_dsm_score_estimation(
    score_net, x, cond=None, cond_mask=None, labels=None, loss_type='mse', gamma=False, all_frames=False
):
    '''
    dsm: denosing score matching.
    '''  
    version = getattr(score_net, 'version', 'SMLD').upper()
    
    if all_frames:
        x = torch.cat([x, cond], dim=1)
        cond = None

    # z, perturbed_x
    if version == "SMLD":
        sigmas = score_net.sigmas
        if labels is None:
            labels = torch.randint(0, len(sigmas), (x.shape[0],), device=x.device)
        used_sigmas = sigmas[labels].reshape(x.shape[0], *([1] * len(x.shape[1:])))
        z = torch.randn_like(x)
        perturbed_x = x + used_sigmas * z
    elif version == "DDPM" or version == "DDIM" or version == "FPNDM":
        alphas = score_net.alphas
        if labels is None:  # time steps
            labels = torch.randint(0, len(alphas), (x.shape[0],), device=x.device)
        used_alphas = alphas[labels].reshape(x.shape[0], *([1] * len(x.shape[1:])))
        if gamma:
            used_k = score_net.k_cum[labels].reshape(x.shape[0], *([1] * len(x.shape[1:]))).repeat(1, *x.shape[1:])
            used_theta = score_net.theta_t[labels].reshape(x.shape[0], *([1] * len(x.shape[1:]))).repeat(1, *x.shape[1:])
            z = Gamma(used_k, 1 / used_theta).sample()
            z = (z - used_k*used_theta) / (1 - used_alphas).sqrt()
        else:
            z = torch.randn_like(x)
        perturbed_x = used_alphas.sqrt() * x + (1 - used_alphas).sqrt() * z
        
        
    # Loss
    if loss_type == 'L1':
        def pow_(x):
            return x.abs()
    elif loss_type == 'mse':
        def pow_(x):
            return 1 / 2. * x.square()
    else:
        raise NotImplementedError

    loss = pow_(
        (z - score_net(perturbed_x, labels, cond=cond, cond_mask=cond_mask)).reshape(len(x), -1)
    ).sum(dim=-1)

    return loss.mean(dim=0)


@MODELS.register_module()
class AnnealDSMScoreDstimation(nn.Module):
    def __init__(self, loss_type='mse', gamma=False, all_frames=False, ):
        super().__init__()
        self.loss_fn = partial(anneal_dsm_score_estimation,
            loss_type=loss_type, gamma=gamma, all_frames=all_frames
        )
    
    def forward(self, score_net, x, cond=None, cond_mask=None, labels=None, ):
        loss = self.loss_fn(
            score_net=score_net, x=x, cond=cond, cond_mask=cond_mask, labels=labels
        )
        return dict(dsm_loss=loss)