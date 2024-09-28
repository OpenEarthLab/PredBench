import torch
import torch.nn as nn
import einops
import numpy as np
import math
import random

from predbench.registry import MODELS
from .lstm_cells import PhyCell, PhyD_ConvLSTM, PhyD_EncoderRNN, K2M
from .rnn_base import RNNBase


@MODELS.register_module()
class PhyDNet(RNNBase):
    def __init__(self,
        input_shape, 
        num_layers, 
        num_hidden, 
        patch_size, 
        sr_size,
        km_lambda = 1.0,
    ):
        super().__init__(input_shape, num_layers, num_hidden, patch_size)
        self.km_lambda = km_lambda  # kernel moment lambda

        sr_size = sr_size if sr_size in [2, 4] else 4
        self.width = self.W_patch // sr_size
        self.height = self.H_patch // sr_size
        
        self.phycell = PhyCell(input_shape=(self.height, self.width), input_dim=64, F_hidden_dims=[49],
                               n_layers=1, kernel_size=(7,7))
        self.convcell = PhyD_ConvLSTM(input_shape=(self.height, self.width), input_dim=64, hidden_dims=[128,128,64],
                                      n_layers=3, kernel_size=(3,3))
        self.encoder = PhyD_EncoderRNN(self.phycell, self.convcell,
                                       in_channel=self.C_img, patch_size=sr_size)
        self.k2m = K2M([7,7])
        
        self.constraints = self._get_constraints()
        self.criterion = nn.MSELoss()

    def _get_constraints(self):
        constraints = torch.zeros((49, 7, 7))
        ind = 0
        for i in range(0, 7):
            for j in range(0, 7):
                constraints[ind,i,j] = 1
                ind +=1
        return constraints 

    def forward(self, patches, teacher_forcing_ratio, target_len):
        input_tensor, target_tensor = patches.split([self.input_len, target_len], dim=1)
        mse_loss = 0
        for ei in range(self.input_len - 1):
            _, _, output_image, _, _ = self.encoder(input_tensor[:,ei,:,:,:], (ei==0))
            mse_loss += self.criterion(output_image, input_tensor[:,ei+1,:,:,:])
            # next_patches.append(output_image)
        
        decoder_input = input_tensor[:,-1,:,:,:]
        predictions = []
        
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        for di in range(target_len):
            _, _, output_image, _, _ = self.encoder(decoder_input)
            target = target_tensor[:,di,:,:,:]
            mse_loss += self.criterion(output_image, target)
            if use_teacher_forcing:
                decoder_input = target
            else:
                decoder_input = output_image
            predictions.append(output_image)
            
        km_loss = 0 # kernel moment loss
        for b in range(0, self.encoder.phycell.cell_list[0].input_dim):
            filters = self.encoder.phycell.cell_list[0].F.conv1.weight[:,b,:,:]
            m = self.k2m(filters.double()).float()
            km_loss += self.criterion(m, self.constraints.to(m.device))
        
        predictions = einops.rearrange(
            torch.stack(predictions, dim=0), 't b c h w -> b t c h w'
        )
        return predictions, dict(mse_loss=mse_loss, km_loss=self.km_lambda*km_loss)
    
    