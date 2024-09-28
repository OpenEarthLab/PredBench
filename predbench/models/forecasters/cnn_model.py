import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from mmengine.model import BaseModel
from predbench.registry import MODELS



@MODELS.register_module()
class CNNModel(BaseModel):
    def __init__(self, data_processor, CNN, loss_fn):
        super().__init__(data_preprocessor=data_processor)
        self.CNN = MODELS.build(CNN)
        self.loss_fn = MODELS.build(loss_fn)
        
        self.input_len = data_processor.input_len
        self.output_len = data_processor.output_len
        self.pred_len = data_processor.pred_len
        
        
    @property
    def data_processor(self):
        return self.data_preprocessor

    def whole_inference(self, input):
        if self.input_len == self.output_len:      
            pred=self.CNN(input)
        elif self.input_len < self.output_len:
            pred = input.clone()
            n_iter = self.output_len // self.input_len
            n_remain = self.output_len % self.input_len
            input_cur = input
            for _ in range(n_iter):
                pred_tmp = self.CNN(input_cur)
                pred = torch.cat([pred, pred_tmp], dim=1)
                input_cur = pred[:, -self.input_len:, ...]
            if n_remain != 0:
                pred_tmp = self.CNN(input_cur)
                pred = torch.cat([pred, pred_tmp[:, :n_remain, ...]], dim=1)
            pred = pred[:, self.input_len:, ...]      
        else:   # self.input_len > out_len
            pred = self.CNN(input)[:, :self.output_len, ...] # b t c h w
        return pred
        
    def forward_train(self, input, target):
        pred = self.whole_inference(input)
        losses = self.loss_fn(pred, target)
        return losses
        
    def forward_inference(self, input, target):
        if self.output_len == self.pred_len:
            pred = self.whole_inference(input)
        elif self.output_len > self.pred_len:
            pred = self.whole_inference(input)[:, :self.pred_len, ...]    # b t c h w
        else:   # self.output_len < self.pred_len
            pred = input.clone()
            n_iter = self.pred_len // self.output_len
            n_remain = self.pred_len % self.output_len
            input_tmp = input
            for _ in range(n_iter):
                pred_tmp = self.whole_inference(input_tmp)
                pred = torch.cat([pred, pred_tmp], dim=1)
                input_tmp = pred[:, -self.input_len:, ...]
            if n_remain != 0:
                pred_tmp = self.whole_inference(input_tmp)
                pred = torch.cat([pred, pred_tmp[:, :n_remain, ...]], dim=1)
            pred = pred[:, self.input_len:, ...]
        return [
            self.data_processor(
                dict(pred=pred.clone(), gt=target.clone()), False, post_process=True
            ),
            dict(pred=pred, gt=target),
        ]
        
    def forward(self, input, target, mode='loss'):        
        if mode == 'loss':
            return self.forward_train(input, target)
        elif mode == 'predict':
            return self.forward_inference(input, target)
        else:
            raise NotImplementedError


