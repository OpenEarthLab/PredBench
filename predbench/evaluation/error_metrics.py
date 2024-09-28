import copy
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from tqdm import tqdm
from functools import partial

import mmengine
import numpy as np
import torch
from mmengine.evaluator import BaseMetric
import einops
from predbench.registry import METRICS


@METRICS.register_module()
class ErrorMetrics(BaseMetric):
    def __init__(
        self, 
        metric_list: Optional[Union[str, Tuple[str]]] = ['mae', 'mse', 'rmse'],
        norm_01: bool = True,
        spatial_norm: bool = False,
        by_frame: bool = False,
        collect_device: str = 'cpu', 
        prefix: str  = 'error', 
        collect_dir: str  = None
    ) -> None:
        super().__init__(collect_device, prefix, collect_dir)
        self.metric_fn = dict(
            mae=self.cal_MAE, mse=self.cal_MSE, rmse=self.cal_RMSE, 
            mape=self.cal_MAPE, wmape=self.cal_wMAPE
        )
        self.metric_list = metric_list
        self.norm_01 = norm_01
        self.spatial_norm = spatial_norm
        self.by_frame = by_frame
        
    
    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        result = dict()
        # print('-'*50)
        # print(torch.max(data_samples[0]['pred']), torch.min(data_samples[0]['pred']), torch.max(data_samples[0]['gt']), torch.min(data_samples[0]['gt']))
        # print(torch.max(data_samples[1]['pred']), torch.min(data_samples[1]['pred']), torch.max(data_samples[1]['gt']), torch.min(data_samples[1]['gt']))
        if self.spatial_norm:   # calculate metric in original space
            pred = data_samples[0]['pred']
            gt = data_samples[0]['gt']
        else:                   # calculate metric in normalized space
            if not self.norm_01:    # from [-1,1] to [0, 1]
                pred = (data_samples[1]['pred'] + 1.) / 2.
                gt = (data_samples[1]['gt'] + 1.) / 2.
            else:
                pred = data_samples[1]['pred']
                gt = data_samples[1]['gt']
        for metric in self.metric_list:
            if not self.by_frame:
                result[metric] = self.metric_fn[metric](pred, gt, spatial_norm=self.spatial_norm)
            else:
                assert len(pred.shape) == 5, f'image shape is {pred.shape}, only video is supported'
                metric_res = []
                for t in range(pred.shape[1]):
                    metric_res.append(
                        self.metric_fn[metric](pred[:,t], gt[:,t], spatial_norm=self.spatial_norm)
                    )
                result[metric] = torch.stack(metric_res)
                
        self.results.append(result)
        # self.gt.append(data_samples[1]['gt'].cpu())
        # self.pred.append(data_samples[1]['pred'].cpu())
        

    def compute_metrics(self, results: list) -> dict:
        
        # np.save('test_metrics/gt_kth_e3dlstm.npy', np.array(torch.cat(self.gt, dim=0)))
        # np.save('test_metrics/pred_kth_e3dlstm.npy', np.array(torch.cat(self.pred, dim=0)))
        metrics = dict()
        for metric in self.metric_list:
            metrics[metric] = torch.stack([result[metric] for result in results]).mean(dim=0)
        return metrics
        
    @staticmethod
    def cal_MAE(pred, gt, spatial_norm=False):
        '''
        gt and pred has the shape of 
            [n t c h w] (video)
            [n c h w] (image)
        '''
        C, H, W = gt.shape[-3: ]
        error = torch.abs(pred-gt)
        if spatial_norm:    # sum(error) / (n*t*c*h*w) = mean(error)
            return torch.mean(error).detach().cpu()
        else:               # sum(error) / (n*t) = mean(error) * (H*W*C)
            return torch.mean(error).detach().cpu() * (C*H*W)
        # if spatial_norm:    # sum(error) / (n*t*c*h*w)
        #     return torch.mean(torch.abs(pred-gt)).detach().cpu()
        # else:               # sum(error) / (n*t)
        #     return torch.mean(
        #         torch.abs(pred-gt), dim=tuple(range(len(pred.shape)-3)) # For video, dim=(0,1). For image, dim=(0).
        #     ).sum().detach().cpu()
        
        
    @staticmethod
    def cal_MSE(pred, gt, spatial_norm=False):
        '''
        gt and pred has the shape of 
            [n t c h w] (video)
            [n c h w] (image)
        '''
        C, H, W = pred.shape[-3: ]
        error = (pred-gt)**2
        if spatial_norm:    # sum(error) / (n*t*c*h*w) = mean(error)
            return torch.mean(error).detach().cpu()
        else:               # sum(error) / (n*t) = mean(error) * (H*W*C)
            return torch.mean(error).detach().cpu() * (C*H*W)
        # if spatial_norm:    # sum(error) / (n*t*c*h*w)
        #     return torch.mean((pred-gt)**2).detach().cpu()
        # else:               # sum(error) / (n*t)
        #     return torch.mean(
        #         (pred-gt)**2, dim=tuple(range(len(pred.shape)-3))   # For video, dim=(0,1). For image, dim=(0).
        #     ).sum().detach().cpu()

    @staticmethod
    def cal_RMSE(pred, gt, spatial_norm=False):
        '''
        gt and pred has the shape of 
            [n t c h w] (video)
            [n c h w] (image)
        '''
        def cal_rmse(pred, gt, spatial_norm):
            '''
            calculate rmse for single image
            '''
            if spatial_norm:    # sqrt(sum(error)/(c*h*w)) = sqrt(mean(error))
                return torch.sqrt(torch.mean((pred-gt)**2)).detach().cpu()
            else:               # sqrt(sum(error))
                return torch.sqrt(torch.sum((pred-gt)**2)).detach().cpu()
        if len(pred.shape) == 5:
            pred = einops.rearrange(pred, 'n t c h w -> (n t) c h w')
            gt = einops.rearrange(gt, 'n t c h w -> (n t) c h w')
        rmse = 0
        for n in range(pred.shape[0]):
            rmse += cal_rmse(pred[n], gt[n], spatial_norm).detach().cpu()
        return rmse / pred.shape[0]

    @staticmethod
    def cal_MAPE(pred, gt, **kwargs):
        '''
        gt and pred has the shape of 
            [n t c h w] (video)
            [n c h w] (image)
        '''
        error = torch.abs(gt-pred) / torch.abs(gt)
        error = torch.where(error.isinf(), torch.full_like(error, 0.), error)
        return torch.nanmean(error).detach().cpu()
        
        
    @staticmethod
    def cal_wMAPE(pred, gt, **kwargs):
        '''
        gt and pred has the shape of 
            [n t c h w] (video)
            [n c h w] (image)
        '''
        # print(
        #     torch.max(gt), torch.min(gt), torch.max(pred), torch.min(pred),
        #     torch.sum(torch.abs(gt-pred)) / torch.sum(torch.abs(gt))
        # )
        return (torch.sum(torch.abs(gt-pred)) / torch.sum(torch.abs(gt))).detach().cpu()
        

