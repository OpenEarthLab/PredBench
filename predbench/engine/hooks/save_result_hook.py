from typing import Dict, Optional, Sequence, Union, Tuple
from mmengine.hooks.hook import DATA_BATCH
from predbench.registry import HOOKS
from mmengine.hooks import Hook

import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import torch
import einops

@HOOKS.register_module()
class SaveResultHook(Hook):
    def __init__(self, max_n_saved=128) -> None:
        super().__init__()
        self.max_n_saved = max_n_saved
        self.n_saved = 0
        self.inputs = []
        self.targets = []
        self.preds = []
            
    
    def after_test_iter(self, runner, batch_idx: int, data_batch: DATA_BATCH = None, outputs: Sequence = None):
        if self.n_saved >= self.max_n_saved:
            return
        gt_input_target = torch.stack(data_batch['imgs'], dim=0).cpu().numpy()  # b (t_in, t_pred) c h w
        pred = outputs[0]['pred'].cpu().numpy() # b t_pred c h w
        B, T_all, C, H, W = gt_input_target.shape
        B, T_pred, C, H, W = pred.shape
        gt_input, gt_target = np.split(gt_input_target, [T_all-T_pred, ], axis=1)
        self.n_saved += B
        
        self.inputs.append(gt_input)
        self.targets.append(gt_target)
        self.preds.append(pred)
        if self.n_saved >= self.max_n_saved:
            save_dir = osp.join(runner._work_dir, 'results')
            if not osp.exists(save_dir): 
                os.mkdir(save_dir)
            np.save(osp.join(save_dir, 'inputs.npy'), np.concatenate(self.inputs, axis=0))
            np.save(osp.join(save_dir, 'targets.npy'), np.concatenate(self.targets, axis=0))
            np.save(osp.join(save_dir, 'preds.npy'), np.concatenate(self.preds, axis=0))
            
            # TODO: remember to delete this !!!
            # raise RuntimeError
        