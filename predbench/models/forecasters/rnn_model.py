import torch
import einops
import numpy as np
import math
from mmengine.model import BaseModel
from predbench.registry import MODELS


@MODELS.register_module()
class RNNModel(BaseModel):
    def __init__(self, 
        data_processor,
        RNN, 
        loss_fn,
        # reverse scheduled sampling
        reverse_scheduled_sampling = False,
        r_sampling_step_1 = 25000,
        r_sampling_step_2 = 50000,
        r_exp_alpha = 5000,
        # scheduled sampling
        scheduled_sampling = True,
        sampling_stop_iter = 50000,
        sampling_start_value = 1.0,
        sampling_changing_rate = 0.00002,
    ):
        super().__init__(data_preprocessor=data_processor)
        
        self.RNN = MODELS.build(RNN)
        self.loss_fn = MODELS.build(loss_fn)
        
        self.input_len = data_processor.input_len
        self.output_len = data_processor.output_len
        self.pred_len = data_processor.pred_len

        self.patch_size = self.RNN.patch_size
        self.H_patch = self.RNN.H_patch
        self.W_patch = self.RNN.W_patch
        self.C_patch = self.RNN.C_patch
        
        # training strategies
        assert not (reverse_scheduled_sampling==True and scheduled_sampling==True)
        # reverse scheduled sampling
        self.reverse_scheduled_sampling = reverse_scheduled_sampling
        self.r_sampling_step_1 = r_sampling_step_1
        self.r_sampling_step_2 = r_sampling_step_2
        self.r_exp_alpha = r_exp_alpha
        # scheduled sampling
        self.scheduled_sampling = scheduled_sampling
        self.sampling_stop_iter = sampling_stop_iter
        self.eta = sampling_start_value # initialization
        self.sampling_changing_rate = sampling_changing_rate
        # teacher forcing (PhyDNet)
        self.teacher_forcing = True if RNN.type in ['PhyDNet'] else False
        
        # number of iterations and epochs
        self._iter = 0
        self._epoch = 0
        
        
    @property
    def data_processor(self):
        return self.data_preprocessor
        

    def forward_train(self, patches):
        if self.teacher_forcing:    # PhyDNet
            teacher_forcing_ratio = np.maximum(0 , 1 - self._epoch * 0.003) 
            results = self.RNN(patches, teacher_forcing_ratio, self.output_len)
            next_frames, phydnet_loss = results
            assert isinstance(phydnet_loss, dict)
            return phydnet_loss
        else:
            if self.reverse_scheduled_sampling:
                mask_gt = self.reverse_scheduled_sampling_fn(patches.shape[0])
            elif self.scheduled_sampling:
                mask_gt = self.scheduled_sampling_fn(patches.shape[0])
            else:
                mask_gt = np.zeros(
                    (patches.shape[0], self.output_len - 1, self.C_patch, self.H_patch, self.W_patch)
                )
            mask_gt = torch.tensor(mask_gt, dtype=patches.dtype, device=patches.device)
            results = self.RNN(patches, mask_gt, self.output_len, self.reverse_scheduled_sampling)
            
            if isinstance(results, torch.Tensor):
                losses = self.loss_fn(results, patches[:, 1:])
            else:
                next_frames, extra_loss = results
                assert isinstance(extra_loss, dict)
                losses = self.loss_fn(next_frames, patches[:, 1:])
                losses.update(extra_loss)
            return losses
    
    def forward_inference(self, patches):
        if self.teacher_forcing:
            teacher_forcing_ratio = 0.0 # inference stage, 
            results = self.RNN(patches, teacher_forcing_ratio, self.pred_len)
        else:
            mask_input_len = 1 if self.reverse_scheduled_sampling else self.input_len
            mask_gt = torch.zeros(
                (
                    patches.shape[0], self.input_len + self.pred_len - mask_input_len - 1,
                    self.C_patch, self.H_patch, self.W_patch
                )
            ).to(patches.device)
            if self.reverse_scheduled_sampling == True:
                mask_gt[:, :self.input_len - 1, :, :] = 1.0
                
            results = self.RNN(patches, mask_gt, self.pred_len, self.reverse_scheduled_sampling)
        
        if isinstance(results, torch.Tensor):
            patches_gen = results
        else:
            patches_gen, _ = results
        imgs_pred = self.patch_to_img(patches_gen[:, -self.pred_len:, ...])
        imgs_gt = self.patch_to_img(patches[:, -self.pred_len:, ...])
        
        return [
            self.data_processor(
                dict(pred=imgs_pred.clone(), gt=imgs_gt.clone()), False, post_process=True
            ),
            dict(pred=imgs_pred, gt=imgs_gt),
        ]
        
    def forward(self, input, target, mode='loss'):
        imgs = torch.cat([input, target], dim=1)
        patches = self.img_to_patch(imgs)
        if mode == 'loss':
            return self.forward_train(patches)
        elif mode == 'predict':
            return self.forward_inference(patches)
        else:
            raise NotImplementedError

    def img_to_patch(self, imgs):
        patches = einops.rearrange(
            imgs, 'b t c (hp p1) (wp p2) -> b t (c p1 p2) hp wp', p1=self.patch_size, p2=self.patch_size
        )
        return patches

    def patch_to_img(self, patches):
        imgs = einops.rearrange(
            patches, 'b t (c p1 p2) hp wp -> b t c (hp p1) (wp p2)', p1=self.patch_size, p2=self.patch_size
        )
        return imgs
        
    def reverse_scheduled_sampling_fn(self, batch_size):
        if self._iter < self.r_sampling_step_1:
            r_eta = 0.5
        elif self._iter < self.r_sampling_step_2:
            r_eta = 1.0 - 0.5 * math.exp(-float(self._iter - self.r_sampling_step_1) / self.r_exp_alpha)
        else:
            r_eta = 1.0

        if self._iter < self.r_sampling_step_1:
            eta = 0.5
        elif self._iter < self.r_sampling_step_2:
            eta = 0.5 - (0.5 / (self.r_sampling_step_2 - self.r_sampling_step_1)) * (self._iter - self.r_sampling_step_1)
        else:
            eta = 0.0
        
        r_random_flip = np.random.random_sample((batch_size, self.input_len - 1))
        r_true_token = (r_random_flip < r_eta)

        random_flip = np.random.random_sample((batch_size, self.output_len - 1))
        true_token = (random_flip < eta)
        mask_gt = np.zeros(
            (
                batch_size, self.input_len + self.output_len - 2,
                self.C_patch, self.H_patch, self.W_patch             
            )
        )
        mask_gt[np.concatenate([r_true_token, true_token], axis=1)] = np.ones(
            (self.C_patch, self.H_patch, self.W_patch)
        )
        return mask_gt
        
    def scheduled_sampling_fn(self, batch_size):
        if not self.scheduled_sampling:
            return 
        if self._iter < self.sampling_stop_iter:
            self.eta -= self.sampling_changing_rate
        else:
            self.eta = 0.0
        
        random_flip = np.random.random_sample(
            (batch_size, self.output_len - 1))
                
        true_token = (random_flip < self.eta)
        mask_gt = np.zeros(
            (batch_size, self.output_len - 1, self.C_patch, self.H_patch, self.W_patch)
        )
        mask_gt[true_token] = np.ones((self.C_patch, self.H_patch, self.W_patch))
        return mask_gt

