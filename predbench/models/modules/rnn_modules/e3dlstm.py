import torch
import torch.nn as nn
import einops
import numpy as np
import math
import torch.nn.functional as F
from predbench.registry import MODELS
from .lstm_cells import Eidetic3DLSTMCell
from .rnn_base import RNNBase

@MODELS.register_module()
class E3DLSTM(RNNBase):
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
        
        self.window_length = 2
        self.window_stride = 1
        cell_list = []
        for i in range(num_layers):
            in_channel = self.C_patch if i == 0 else num_hidden[i - 1]
            cell_list.append(
                Eidetic3DLSTMCell(
                    in_channel, num_hidden[i], self.window_length, self.H_patch, self.W_patch, filter_size, stride, layer_norm
                )
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv3d(
            num_hidden[num_layers - 1], self.C_patch, kernel_size=(self.window_length, 1, 1),
            stride=(self.window_length, 1, 1), padding=0, bias=False
        )
        
        
    def forward(self, patches, mask_gt, target_len, reverse_scheduled_sampling):
        batch_size=patches.shape[0]
        next_patches = []
        h_t = []
        c_t = []
        c_history = []
        input_list = []

        for t in range(self.window_length - 1):
            input_list.append(
                torch.zeros_like(patches[:, 0]))

        for i in range(self.num_layers):
            zeros = torch.zeros(
                [batch_size, self.num_hidden[i], self.window_length, self.H_patch, self.W_patch]
            ).to(patches.device)
            h_t.append(zeros)
            c_t.append(zeros)
            c_history.append(zeros)

        memory = torch.zeros(
            [batch_size, self.num_hidden[0], self.window_length, self.H_patch, self.W_patch]
        ).to(patches.device)

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

            input_list.append(net)

            if t % (self.window_length - self.window_stride) == 0:
                net = torch.stack(input_list[t:], dim=0)
                net = net.permute(1, 2, 0, 3, 4).contiguous()

            for i in range(self.num_layers):
                if t == 0:
                    c_history[i] = c_t[i]
                else:
                    c_history[i] = torch.cat((c_history[i], c_t[i]), 1)
                
                input = net if i == 0 else h_t[i-1]
                h_t[i], c_t[i], memory = self.cell_list[i](input, h_t[i], c_t[i], memory, c_history[i])

            x_gen = self.conv_last(h_t[self.num_layers - 1]).squeeze(2)
            next_patches.append(x_gen)

        next_patches = einops.rearrange(
            torch.stack(next_patches, dim=0), 't b c h w -> b t c h w'
        )

        return next_patches

