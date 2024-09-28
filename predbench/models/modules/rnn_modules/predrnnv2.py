import torch
import torch.nn as nn
import einops
import numpy as np
import math
import torch.nn.functional as F

from predbench.registry import MODELS
from .lstm_cells import SpatioTemporalLSTMCellv2
from .rnn_base import RNNBase


@MODELS.register_module()
class PredRNNv2(RNNBase):
    def __init__(self,
        input_shape, 
        num_layers, 
        num_hidden, 
        patch_size, 
        filter_size, 
        stride, 
        layer_norm, 
        decouple_beta = 0.1,
    ):
        super().__init__(input_shape, num_layers, num_hidden, patch_size)
        self.decouple_beta = decouple_beta
        cell_list = []
        for i in range(num_layers):
            in_channel = self.C_patch if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCellv2(in_channel, num_hidden[i], self.H_patch, self.W_patch, filter_size, stride, layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(
            num_hidden[num_layers - 1], self.C_patch, kernel_size=1, stride=1, padding=0, bias=False
        )
        # shared adapter
        adapter_num_hidden = num_hidden[0]
        self.adapter = nn.Conv2d(
            adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False
        )

    def forward(self, patches, mask_gt, target_len, reverse_scheduled_sampling):
        batch_size = patches.shape[0]
        next_patches = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []

        decouple_loss = []

        for i in range(self.num_layers):
            zeros = torch.zeros(
                [batch_size, self.num_hidden[i], self.H_patch, self.W_patch]
            ).to(patches.device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        memory = torch.zeros(
            [batch_size, self.num_hidden[0], self.H_patch, self.W_patch]
        ).to(patches.device)
        
        for t in range(self.input_len + target_len - 1):

            if reverse_scheduled_sampling == True:
                # reverse schedule sampling
                if t == 0:
                    net = patches[:, t]
                else:
                    net = mask_gt[:, t - 1] * patches[:, t] + (1 - mask_gt[:, t - 1]) * x_gen
            else:
                # schedule sampling
                if t < self.input_len:
                    net = patches[:, t]
                else:
                    net = mask_gt[:, t - self.input_len] * patches[:, t] + \
                          (1 - mask_gt[:, t - self.input_len]) * x_gen

            h_t[0], c_t[0], memory, delta_c, delta_m = \
                self.cell_list[0](net, h_t[0], c_t[0], memory)
            delta_c_list[0] = F.normalize(
                self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
            delta_m_list[0] = F.normalize(
                self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = \
                    self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                delta_c_list[i] = F.normalize(
                    self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[i] = F.normalize(
                    self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_patches.append(x_gen)

            # decoupling loss
            for i in range(0, self.num_layers):
                decouple_loss.append(
                    torch.mean(
                        torch.abs(torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2))
                    )
                )
        next_patches = einops.rearrange(
            torch.stack(next_patches, dim=0), 't b c h w -> b t c h w'
        )
        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        return next_patches, dict(decouple_loss=self.decouple_beta*decouple_loss)
