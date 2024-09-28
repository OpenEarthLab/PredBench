import torch
import torch.nn as nn
import einops
import numpy as np
import math

from predbench.registry import MODELS
from .lstm_cells import MAUCell
from .rnn_base import RNNBase



@MODELS.register_module()
class MAU(RNNBase):
    def __init__(self,
        input_shape, 
        num_layers, 
        num_hidden, 
        patch_size, 
        filter_size, 
        stride,
        sr_size,
        tau, 
        cell_mode,
        model_mode,
    ):
        super().__init__(input_shape, num_layers, num_hidden, patch_size)
        
        self.tau = tau
        self.cell_mode = cell_mode
        self.model_mode = model_mode
        assert self.model_mode in ['recall', 'normal']
        
        cell_list = []

        self.width = self.W_patch // sr_size
        self.height = self.H_patch // sr_size
        
        for i in range(num_layers):
            in_channel = num_hidden[i - 1]
            cell_list.append(
                MAUCell(in_channel, num_hidden[i], self.height, self.width, filter_size,
                        stride, self.tau, self.cell_mode)
            )
        self.cell_list = nn.ModuleList(cell_list)

        # Encoder
        n = int(math.log2(sr_size))
        encoders = []
        encoder = nn.Sequential()
        encoder.add_module(name='encoder_t_conv{0}'.format(-1),
                           module=nn.Conv2d(in_channels=self.C_patch,
                                            out_channels=self.num_hidden[0],
                                            stride=1,
                                            padding=0,
                                            kernel_size=1))
        encoder.add_module(name='relu_t_{0}'.format(-1),
                           module=nn.LeakyReLU(0.2))
        encoders.append(encoder)
        for i in range(n):
            encoder = nn.Sequential()
            encoder.add_module(name='encoder_t{0}'.format(i),
                               module=nn.Conv2d(in_channels=self.num_hidden[0],
                                                out_channels=self.num_hidden[0],
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                kernel_size=(3, 3)
                                                ))
            encoder.add_module(name='encoder_t_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)
        
        

        # Decoder
        decoders = []

        for i in range(n - 1):
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{0}'.format(i),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoder.add_module(name='c_decoder_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            decoders.append(decoder)

        if n > 0:
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{0}'.format(n - 1),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)

        self.srcnn = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], self.C_patch, kernel_size=1, stride=1, padding=0)
        )
        # self.merge = nn.Conv2d(
        #     self.num_hidden[-1] * 2, self.num_hidden[-1], kernel_size=1, stride=1, padding=0
        # )
        # self.conv_last_sr = nn.Conv2d(
        #     self.C_patch * 2, self.C_patch, kernel_size=1, stride=1, padding=0
        # )

    def forward(self, patches, mask_gt, target_len, reverse_scheduled_sampling):
        batch_size = patches.shape[0]
        next_patches = []
        T_t = []
        T_pre = []
        S_pre = []
        x_gen = None
        for layer_idx in range(self.num_layers):
            tmp_t = []
            tmp_s = []
            if layer_idx == 0:
                in_channel = self.num_hidden[layer_idx]
            else:
                in_channel = self.num_hidden[layer_idx - 1]
            for i in range(self.tau):
                tmp_t.append(torch.zeros(
                    [batch_size, in_channel, self.height, self.width]).to(patches.device))
                tmp_s.append(torch.zeros(
                    [batch_size, in_channel, self.height, self.width]).to(patches.device))
            T_pre.append(tmp_t)
            S_pre.append(tmp_s)

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
                    time_diff = t - self.input_len
                net = mask_gt[:, time_diff] * patches[:, t] + (1 - mask_gt[:, time_diff]) * x_gen
            
            frames_feature = net
            frames_feature_encoded = []
            
            for i in range(len(self.encoders)):
                frames_feature = self.encoders[i](frames_feature)
                frames_feature_encoded.append(frames_feature)
            if t == 0:
                for i in range(self.num_layers):
                    zeros = torch.zeros(
                        [batch_size, self.num_hidden[i], self.height, self.width]).to(patches.device)
                    T_t.append(zeros)
            S_t = frames_feature
            for i in range(self.num_layers):
                t_att = T_pre[i][-self.tau:]
                t_att = torch.stack(t_att, dim=0)
                s_att = S_pre[i][-self.tau:]
                s_att = torch.stack(s_att, dim=0)
                S_pre[i].append(S_t)
                T_t[i], S_t = self.cell_list[i](T_t[i], S_t, t_att, s_att)
                T_pre[i].append(T_t[i])
            out = S_t

            for i in range(len(self.decoders)):
                out = self.decoders[i](out)
                if self.model_mode == 'recall':
                    out = out + frames_feature_encoded[-2 - i]

            x_gen = self.srcnn(out)
            next_patches.append(x_gen)
        next_patches = einops.rearrange(
            torch.stack(next_patches, dim=0), 't b c h w -> b t c h w'
        )
        return next_patches
