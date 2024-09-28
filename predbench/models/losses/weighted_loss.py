import torch.nn as nn
from typing import List
import torch.nn.functional as F
import torch

from predbench.registry import MODELS


class DiffDivReg(nn.Module):
    def __init__(self, tau=0.1, eps=1e-12):
        super().__init__()
        self.tau = tau
        self.eps = eps
        
    def forward(self, pred, gt):
        B, T, C = pred.shape[:3]
        if T <= 2:  return torch.tensor(0., device=gt.device)
        gap_pred_y = (pred[:, 1:] - pred[:, :-1]).reshape(B, T-1, -1)
        gap_batch_y = (gt[:, 1:] - gt[:, :-1]).reshape(B, T-1, -1)
        softmax_gap_p = F.softmax(gap_pred_y / self.tau, -1)
        softmax_gap_b = F.softmax(gap_batch_y / self.tau, -1)
        loss_gap = softmax_gap_p * \
            torch.log(softmax_gap_p / (softmax_gap_b + self.eps) + self.eps)
        return loss_gap.mean()

loss_functions_dict = dict(
    mse_loss = nn.MSELoss(),  # L2 or Frobenius 
    L1_loss = nn.L1Loss(),
    ddr_loss = DiffDivReg(),    # Differential Divergence Regularization
)


@MODELS.register_module()
class WeightedLoss(nn.Module):
    def __init__(self,
        loss_functions_list: List = ['mse_loss', ],
        loss_weights_list: List[int] = [1.0, ],
    ):
        super().__init__()
        self.loss_functions = dict()
        assert len(loss_functions_list) == len(loss_weights_list)
        for loss_fn in loss_functions_list:
            self.loss_functions[loss_fn] = loss_functions_dict[loss_fn]
        self.loss_weights = loss_weights_list
    def forward(self, pred, gt):
        losses = dict()
        for idx, (k, loss_fn) in enumerate(self.loss_functions.items()):
            losses[k] = self.loss_weights[idx] * loss_fn(pred, gt)
        return losses

