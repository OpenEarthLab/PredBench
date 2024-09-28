import torch
import einops
from torch.distributions.gamma import Gamma
from math import ceil
from mmengine.model import BaseModel
from predbench.registry import MODELS


@MODELS.register_module()
class DiffusionModel(BaseModel):
    def __init__(self, 
        data_processor, loss_fn, sampler, score_net, prob_cond_mask, gamma=False, 
    ):
        super().__init__(data_preprocessor=data_processor)
        self.score_net = MODELS.build(score_net)
        self.loss_fn = MODELS.build(loss_fn)
        self.sampler = MODELS.build(sampler)
        
        self.input_len = data_processor.input_len
        self.output_len = data_processor.output_len
        self.pred_len = data_processor.pred_len
        
        self.prob_cond_mask = prob_cond_mask        
        self.gamma = gamma

    @property
    def data_processor(self):
        return self.data_preprocessor

    def forward_train(self, input, target):
        input = einops.rearrange(input, 'b t c h w -> b (t c) h w')
        target = einops.rearrange(target, 'b t c h w -> b (t c) h w')
        if self.prob_cond_mask > 0.0:
            cond_mask = einops.rearrange(
                (torch.rand(input.shape[0], device=input.device) > self.prob_cond_mask), 
                'b -> b 1 1 1'
            )
            input = input * cond_mask
            cond_mask = cond_mask.to(torch.int32)
        else:
            cond_mask = None
        losses = self.loss_fn(
            self.score_net, target, cond=input, cond_mask=cond_mask, labels=None, 
        )
        assert isinstance(losses, dict)
        return losses
    
    def forward_inference(self, input, target):
                
        B, T, C, H, W = input.shape
        cond = einops.rearrange(input, 'b t c h w -> b (t c) h w')
        
        init_samples_shape = (B, C*self.output_len, H, W)
        n_iter_frames = ceil(self.pred_len / self.output_len)
        
        pred_samples = []
        for i_frame in range(n_iter_frames):
            if self.gamma:  
                used_k, used_theta = self.score_net.k_cum[0], self.score_net.theta_t[0]
                z = Gamma(
                    torch.full(init_samples_shape, used_k), torch.full(init_samples_shape, 1 / used_theta)
                ).sample().to(cond.device)
                z = z - used_k * used_theta
            else:
                z = torch.randn(init_samples_shape, device=cond.device)
            init_samples = z
            
            # Generate samples
            gen_samples = self.sampler(init_samples, self.score_net, cond=cond)
            gen_samples = gen_samples[-1].reshape(B, C*self.output_len, H, W)
            pred_samples.append(gen_samples)
            
            # update cond
            cond = torch.cat([cond, gen_samples], dim=1)[:, -C*self.input_len:, ...]            

        pred = torch.cat(pred_samples, dim=1)[:, :C*self.pred_len]
        pred = einops.rearrange(pred, 'b (t c) h w -> b t c h w', c = C)
        
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






