import torch.nn as nn
from typing import List
import torch.nn.functional as F
import torch

from predbench.registry import MODELS


@MODELS.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index, label_smoothing, loss_weight=1.0):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing)
        self.loss_weight = loss_weight
        
    def forward(self, logits, labels):
        losses = dict(ce_loss = self.loss_fn(logits, labels))
        return losses

