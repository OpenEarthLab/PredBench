import torch
import torch.nn as nn
import einops
import numpy as np
import math

from predbench.registry import MODELS
from .lstm_cells import CausalLSTMCell, GHU
from .rnn_base import RNNBase



@MODELS.register_module()
class PredRNNpp(RNNBase):
    def __init__(self,
        input_shape, 
        num_layers, 
        num_hidden, 
        patch_size, 
        filter_size, 
        stride, 
        layer_norm, 
        
    ):
        super().__init__(input_shape, num_layers, num_hidden, patch_size)
        cell_list = []
        self.gradient_highway = GHU(
            num_hidden[0], num_hidden[0], self.H_patch, self.W_patch, filter_size, stride, layer_norm
        )

        for i in range(num_layers):
            in_channel = self.C_patch if i == 0 else num_hidden[i - 1]
            cell_list.append(
                CausalLSTMCell(in_channel, num_hidden[i], self.H_patch, self.W_patch, filter_size, stride, layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(
            num_hidden[num_layers - 1], self.C_patch, kernel_size=1, stride=1, padding=0, bias=False
        )
        


    def forward(self, patches, mask_gt, target_len, reverse_scheduled_sampling):
        batch_size = patches.shape[0]
        next_patches = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros(
                [batch_size, self.num_hidden[i], self.H_patch, self.W_patch]
            ).to(patches.device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros(
            [batch_size, self.num_hidden[0], self.H_patch, self.W_patch]
        ).to(patches.device)
        z_t = None

        for t in range(self.input_len + target_len - 1):
            # reverse schedule sampling
            if reverse_scheduled_sampling == True:
                if t == 0:
                    net = patches[:, t]
                else:
                    net = mask_gt[:, t - 1] * patches[:, t] + (1 - mask_gt[:, t - 1]) * x_gen
            else:
                if t < self.input_len:
                    net = patches[:, t]
                else:
                    net = mask_gt[:, t - self.input_len] * patches[:, t] + \
                          (1 - mask_gt[:, t - self.input_len]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)
            z_t = self.gradient_highway(h_t[0], z_t)
            h_t[1], c_t[1], memory = self.cell_list[1](z_t, h_t[1], c_t[1], memory)

            for i in range(2, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_patches.append(x_gen)

        next_patches = einops.rearrange(
            torch.stack(next_patches, dim=0), 't b c h w -> b t c h w'
        )
        return next_patches

