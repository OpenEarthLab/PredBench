from typing import Dict, Optional, Sequence, Union, Tuple
from mmengine.hooks.hook import DATA_BATCH
from predbench.registry import HOOKS
from mmengine.hooks import Hook

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import einops
from PIL import Image
import cv2
from .viz_heatmap_data import (
    visualize_TaxiBJ, visualize_Traffic4cast2021, 
    visualize_SEVIR, visualize_ENSO, visualize_WeatherBench,
)


@HOOKS.register_module()
class VizVideoHook(Hook):
    def __init__(self, data_type=None, max_n_viz=16, viz_stages=['test']) -> None:
        super().__init__()
        if data_type != None:
            assert isinstance(data_type, str)
            data_type = data_type.lower()
        self.data_type = data_type
        self.max_n_viz = max_n_viz
        self.viz_stages = viz_stages
        
        
    def after_val_iter(self, runner, batch_idx: int, data_batch: DATA_BATCH = None, outputs: Sequence = None) -> None:
        if batch_idx == 0 and 'val' in self.viz_stages:
            self.viz_fn(runner._work_dir, runner._train_loop._iter, data_batch, outputs)
        else:
            return
    
    def after_test_iter(self, runner, batch_idx: int, data_batch: DATA_BATCH = None, outputs: Sequence = None):
        if batch_idx == 0 and 'test' in self.viz_stages:
            self.viz_fn(runner._work_dir, 'test_iter', data_batch, outputs)
        else:
            return
    
    def viz_fn(self, work_dir, iter, data_batch: DATA_BATCH = None, outputs: Sequence = None):
        # obtain original input-output and prediction
        gt_input_target = torch.stack(data_batch['imgs'], dim=0).cpu().numpy()  # b (t_in, t_pred) c h w
        pred = outputs[0]['pred'].cpu().numpy() # b t_pred c h w
        # get visualization number
        n_viz = np.minimum(pred.shape[0], self.max_n_viz)
        gt_input_target = gt_input_target[: n_viz]
        pred = pred[: n_viz]
        B, T_all, C, H, W = gt_input_target.shape
        B, T_pred, C, H, W = pred.shape
        
        
        # data pre-process
        if self.data_type in ['enso', 'sevir', 'weatherbench', 'taxibj']:
            pass
        elif self.data_type in ['traffic4cast2021']:
            # https://github.com/iarai/NeurIPS2021-traffic4cast/blob/65f8078e51ac8e8bbee2c427ceb947e246d4002d/metrics/visualization.py
            # The sum over all channels in the last dimension is displayed.
            gt_input_target = np.sum(gt_input_target, axis=2, keepdims=True)
            pred = np.sum(pred, axis=2, keepdims=True)
        else:
            pred = np.clip(pred, 0., 255.)
        
        
        # visualization
        if self.data_type == 'taxibj':
            gt_input, gt_target = np.split(gt_input_target, [T_all-T_pred, ], axis=1)
            gt_input = gt_input[:,:,0,...]
            gt_target = gt_target[:,:,0,...]
            pred = pred[:,:,0,...]
            # gt_input = einops.rearrange(gt_input, 'b t 1 h w -> b t h w')
            # gt_target = einops.rearrange(gt_target, 'b t 1 h w -> b t h w')
            # pred = einops.rearrange(pred, 'b t 1 h w -> b t h w')
            for b in range(n_viz):
                visualize_TaxiBJ(
                    gt_input[b], gt_target[b], [pred[b], ], [], 
                    f'{work_dir}/viz_{iter}_{b}.png'
                )
        elif self.data_type == 'traffic4cast2021':
            gt_input, gt_target = np.split(gt_input_target, [T_all-T_pred, ], axis=1)
            gt_input = einops.rearrange(gt_input, 'b t 1 h w -> b t h w')
            gt_target = einops.rearrange(gt_target, 'b t 1 h w -> b t h w')
            pred = einops.rearrange(pred, 'b t 1 h w -> b t h w')
            for b in range(n_viz):
                visualize_Traffic4cast2021(
                    gt_input[b], gt_target[b], [pred[b], ], [], 
                    f'{work_dir}/viz_{iter}_{b}.png'
                )
        elif self.data_type == 'sevir':
            gt_input, gt_target = np.split(gt_input_target, [T_all-T_pred, ], axis=1)
            gt_input = einops.rearrange(gt_input, 'b t 1 h w -> b t h w')
            gt_target = einops.rearrange(gt_target, 'b t 1 h w -> b t h w')
            pred = einops.rearrange(pred, 'b t 1 h w -> b t h w')
            for b in range(n_viz):
                visualize_SEVIR(
                    gt_input[b], gt_target[b], [pred[b], ], [], 
                    f'{work_dir}/viz_{iter}_{b}.png'
                )
        elif self.data_type == 'enso':
            gt_input, gt_target = np.split(gt_input_target, [T_all-T_pred, ], axis=1)
            gt_input = einops.rearrange(gt_input, 'b t 1 h w -> b t h w')
            gt_target = einops.rearrange(gt_target, 'b t 1 h w -> b t h w')
            pred = einops.rearrange(pred, 'b t 1 h w -> b t h w')
            for b in range(n_viz):
                visualize_ENSO(
                    gt_input[b], gt_target[b], [pred[b], ], [], 
                    f'{work_dir}/viz_{iter}_{b}.png'
                )
        elif self.data_type == 'weatherbench':
            gt_input, gt_target = np.split(gt_input_target, [T_all-T_pred, ], axis=1)
            gt_input_single = gt_input[:, :, 11]
            gt_target_single = gt_target[:, :, 11]
            pred_single = pred[:, :, 11]
            for b in range(n_viz):
                visualize_WeatherBench(
                    gt_input_single[b], gt_target_single[b], [pred_single[b], ], [], 
                    f'{work_dir}/viz_Z500_{iter}_{b}.png'
                )
            gt_input_single = gt_input[:, :, 27]
            gt_target_single = gt_target[:, :, 27]
            pred_single = pred[:, :, 27]
            for b in range(n_viz):
                visualize_WeatherBench(
                    gt_input_single[b], gt_target_single[b], [pred_single[b], ], [], 
                    f'{work_dir}/viz_T850_{iter}_{b}.png'
                )
        else:
            results = np.zeros((2*B, T_all, C, H, W))
            results[0::2,...] = gt_input_target
            results[1::2, -T_pred:, ...] = pred
            results = results.astype(np.uint8)
            results = einops.rearrange(results, 'b t c h w -> (b h) (t w) c')
            cv2.imwrite(
                f'{work_dir}/viz_{iter}.png', cv2.cvtColor(results, cv2.COLOR_RGB2BGR)
            )