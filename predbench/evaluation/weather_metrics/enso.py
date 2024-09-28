import torch
import einops
import numpy as np
from typing import Optional, Union
from configs._base_.datasets.enso import NINO_WINDOW_T


def sst_to_nino(sst_pred: torch.Tensor, sst_gt: torch.Tensor, detach: bool = True):
    '''
    Parameters
    ----------
    sst:    torch.Tensor
        Shape = (N, T, H, W)

    Returns
    -------
    nino_index: torch.Tensor
        Shape = (N, T-NINO_WINDOW_T+1)
    '''
    def _sst_to_nino(sst: torch.Tensor, detach: bool = True):
        if detach:
            nino_index = sst.detach()
        else:
            nino_index = sst
        nino_index = nino_index[:, :, 10:13, 19:30].mean(dim=[2, 3])  # (N, 26)
        nino_index = nino_index.unfold(dimension=1, size=NINO_WINDOW_T, step=1).mean(dim=2)  # (N, 24)
        return nino_index
    sst_pred = einops.rearrange(sst_pred, 'n t 1 h w -> n t h w')
    sst_gt = einops.rearrange(sst_gt, 'n t 1 h w -> n t h w')
    return torch.stack([_sst_to_nino(sst_pred), _sst_to_nino(sst_gt)], dim=0)


def compute_enso_score(
        y_pred, y_true,
        acc_weight: Optional[Union[str, np.ndarray, torch.Tensor]] = None):
    r"""

    Parameters
    ----------
    y_pred: torch.Tensor
    y_true: torch.Tensor
        shape: [n t]
    acc_weight: Optional[Union[str, np.ndarray, torch.Tensor]]
        None:   not used
        default:    use default acc_weight specified at https://tianchi.aliyun.com/competition/entrance/531871/information
        np.ndarray: custom weights

    Returns
    -------
    acc
    rmse
    """
    pred = y_pred - y_pred.mean(dim=0, keepdim=True)  # (N, 24)
    true = y_true - y_true.mean(dim=0, keepdim=True)  # (N, 24)
    cor = (pred * true).sum(dim=0) / (torch.sqrt(torch.sum(pred**2, dim=0) * torch.sum(true**2, dim=0)) + 1e-6)

    if acc_weight is None:
        acc = cor.sum()
    else:
        nino_out_len = y_true.shape[-1]
        if acc_weight == "default":
            acc_weight = torch.tensor(
                [1.5] * 4 + [2] * 7 + [3] * 7 + [4] * (nino_out_len - 18)
            )[:nino_out_len] * torch.log(torch.arange(nino_out_len) + 1)
        elif isinstance(acc_weight, np.ndarray):
            acc_weight = torch.from_numpy(acc_weight[:nino_out_len])
        elif isinstance(acc_weight, torch.Tensor):
            acc_weight = acc_weight[:nino_out_len]
        else:
            raise ValueError(f"Invalid acc_weight {acc_weight}!")
        acc_weight = acc_weight.to(y_pred)
        acc = (acc_weight * cor).sum()
    rmse = torch.mean((y_pred - y_true)**2, dim=0).sqrt().sum()
    return acc / y_pred.shape[-1], rmse / y_pred.shape[-1]
