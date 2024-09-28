import torch.nn as nn
import torch
from abc import abstractmethod, ABCMeta

class RNNBase(nn.Module, metaclass=ABCMeta):
    def __init__(self, 
        input_shape, 
        num_layers, 
        num_hidden, 
        patch_size, 
    ):
        super().__init__()
        
        self.input_len, self.C_img, self.H_img, self.W_img = input_shape
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.patch_size = patch_size

        self.H_patch = self.H_img // patch_size
        self.W_patch = self.W_img // patch_size
        self.C_patch = patch_size * patch_size * self.C_img
        

    @abstractmethod
    def forward(self):
        pass