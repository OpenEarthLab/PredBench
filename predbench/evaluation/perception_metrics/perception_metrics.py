from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmengine.evaluator import BaseMetric

from predbench.registry import METRICS
from .lpips import cal_LPIPS
from .fvd import FVDFeatureExtractor, compute_FVD
from .fid import FIDFeatureExtractor, compute_FID
from .isc import ISProbaExtractor, compute_IS


@METRICS.register_module()
class PerceptionMetrics(BaseMetric):
    def __init__(
        self, 
        metric_list: Optional[Union[str, Tuple[str]]] = ['lpips', 'fvd'],
        norm_01: bool = True,
        by_frame: bool = False,
        lpips_net: str = 'alex', # net in ['alex', 'squeeze', 'vgg']
        fvd_detector: str = 'i3d_400',  # detector in ['i3d_jit', 'i3d_400', 'i3d_600']
        collect_device: str = 'cpu', 
        prefix: str  = 'difference', 
        collect_dir: str  = None
    ) -> None:
        super().__init__(collect_device, prefix, collect_dir)
        use_gpu = True if collect_device == 'gpu' else False
        self.metric_fn = {
            'lpips': cal_LPIPS(net=lpips_net, use_gpu=use_gpu),
            'fvd': FVDFeatureExtractor(i3d_type=fvd_detector, use_gpu=use_gpu),
            'fid': FIDFeatureExtractor(dims=2048, use_gpu=use_gpu),
            'is': ISProbaExtractor(use_gpu=use_gpu)
        }
        self.metric_list = metric_list
        self.norm_01 = norm_01
        self.by_frame = by_frame
        

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        result = dict()
        if self.norm_01:
            pred = data_samples[1]['pred'] * 2.0 - 1.0
            gt = data_samples[1]['gt'] * 2.0 - 1.0
        else:
            pred = data_samples[1]['pred']
            gt = data_samples[1]['gt']
            
        for metric in self.metric_list:
            if not self.by_frame:
                result[metric] = self.metric_fn[metric](pred, gt)
            else:   # actually, only lpips is supported
                assert len(pred.shape) == 5, f'image shape is {pred.shape}, only video is supported'
                metric_res = []
                for t in range(pred.shape[1]):
                    if metric == 'fvd':
                        metric_res.append(
                        self.metric_fn[metric](pred[:, :t], gt[:, :t])
                        )
                    else:
                        metric_res.append(
                            self.metric_fn[metric](pred[:,t], gt[:,t])
                        )
                result[metric] = torch.stack(metric_res)
        self.results.append(result)
        

    def compute_metrics(self, results: list) -> dict:
        metrics = dict()
        for metric in self.metric_list:
            if metric == 'lpips':
                results_tensor = torch.stack([result[metric] for result in results])
                metrics[metric] = results_tensor.mean(dim=0)
            elif metric == 'fvd':
                results_tensor = torch.cat([result[metric] for result in results], dim=-2)
                if self.by_frame:
                    metric_fvd = []
                    for result_tensor in results_tensor:
                        feats_fake, feats_real = result_tensor[0], result_tensor[1]
                        metric_fvd.append(compute_FVD(feats_fake.numpy(), feats_real.numpy()))
                    metrics[metric] = torch.tensor(metric_fvd)
                else: 
                    feats_fake, feats_real = results_tensor[0], results_tensor[1]
                    metrics[metric] = compute_FVD(feats_fake.numpy(), feats_real.numpy())
            elif metric == 'fid':
                results_tensor = torch.cat([result[metric] for result in results], dim=1)
                act_fake, act_real = results_tensor[0], results_tensor[1]
                metrics[metric] = compute_FID(act_fake.numpy(), act_real.numpy())
            elif metric == 'is':
                results_tensor = torch.cat([result[metric] for result in results], dim=0)
                metrics[metric] = compute_IS(results_tensor.numpy(), splits=1)
            else:
                raise NotImplementedError
        return metrics
        
        