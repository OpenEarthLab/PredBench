from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
from mmengine.evaluator import BaseMetric
import einops
from skimage.metrics import structural_similarity as cal_ssim
import numpy as np
from predbench.registry import METRICS



@METRICS.register_module()
class SimilarityMetrics(BaseMetric):
    def __init__(
        self, 
        is_img: bool = True,
        metric_list: Optional[Union[str, Tuple[str]]] = ['ssim', 'psnr', 'snr'],
        norm_01: bool = True,
        by_frame: bool = False,
        collect_device: str = 'cpu', 
        prefix: str  = 'similarity', 
        collect_dir: str  = None
    ) -> None:
        super().__init__(collect_device, prefix, collect_dir)
        self.metric_fn = dict(ssim=self.cal_SSIM, psnr=self.cal_PSNR, snr=self.cal_SNR) 
        self.is_img = is_img
        self.metric_list = metric_list
        self.norm_01 = norm_01
        self.by_frame = by_frame
        
    
    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        result = dict()
        if not self.norm_01:    # from [-1,1] to [0, 1]
            pred = (data_samples[1]['pred'] + 1.) / 2.
            gt = (data_samples[1]['gt'] + 1.) / 2.
        else:
            pred = data_samples[1]['pred']
            gt = data_samples[1]['gt']
            
        for metric in self.metric_list:
            if not self.by_frame:
                result[metric] = self.metric_fn[metric](pred, gt, is_img=self.is_img)
            else:
                assert len(pred.shape) == 5, f'image shape is {pred.shape}, only video is supported'
                metric_res = []
                for t in range(pred.shape[1]):
                    metric_res.append(
                        self.metric_fn[metric](pred[:,t], gt[:,t], is_img=self.is_img)
                    )
                result[metric] = torch.stack(metric_res)
            
        self.results.append(result)
        

    def compute_metrics(self, results: list) -> dict:
        metrics = dict()
        for metric in self.metric_list:
            result_tensor = torch.stack([result[metric] for result in results], dim=0)
            result_tensor = torch.where(
                result_tensor.isinf(), torch.full_like(result_tensor, 1.), result_tensor
            )
            metrics[metric] = result_tensor.nanmean(dim=0)
        
        return metrics
        
    
    
    
    @staticmethod
    def cal_SNR(pred, gt, is_img=True):
        """Signal-to-Noise Ratio.

        Ref: https://en.wikipedia.org/wiki/Signal-to-noise_ratio
        
        gt and pred has the shape of 
            [n t c h w] (video)
            [n c h w] (image)
        """
        def cal_snr(gt, pred):
            signal = ((gt)**2).mean()
            noise = ((gt - pred)**2).mean()
            return 10. * torch.log10(signal / noise).detach().cpu()
        if is_img:
            pred = torch.maximum(pred, torch.min(gt))
            pred = torch.minimum(pred, torch.max(gt))
        if len(pred.shape) == 5:
            pred = einops.rearrange(pred, 'n t c h w -> (n t) c h w')
            gt = einops.rearrange(gt, 'n t c h w -> (n t) c h w')
        snr = 0
        for n in range(pred.shape[0]):
            snr += cal_snr(pred[n], gt[n]).detach().cpu()
        return snr / pred.shape[0]

    
    @staticmethod
    def cal_SSIM(pred, true):
        '''
        gt and pred has the shape of 
            [n t c h w] (video)
            [n c h w] (image)
        '''
        if len(pred.shape) == 5:
            pred = einops.rearrange(pred, 'n t c h w -> (n t) c h w')
            true = einops.rearrange(true, 'n t c h w -> (n t) c h w')
        ssim_v = 0
        for n in range(pred.shape[0]):
            ssim_v += cal_ssim(pred[n], true[n], channel_axis=0)
        return ssim_v / pred.shape[0]
        
        
    @staticmethod
    def cal_PSNR(pred, true):
        '''
        gt and pred has the shape of 
            [n t c h w] (video)
            [n c h w] (image)
            
        Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        '''
        def PSNR(pred, true, min_max_norm=True):
            mse = np.mean((pred.astype(np.float32) - true.astype(np.float32))**2)
            if mse == 0:
                return float('inf')
            else:
                if min_max_norm:  # [0, 1] normalized by min and max
                    return 20. * np.log10(1. / np.sqrt(mse))  # i.e., -10. * np.log10(mse)
                else:
                    return 20. * np.log10(255. / np.sqrt(mse))  # (0, 255)
        if len(pred.shape) == 5:
            pred = einops.rearrange(pred, 'n t c h w -> (n t) c h w')
            true = einops.rearrange(true, 'n t c h w -> (n t) c h w')
        psnr_v = 0
        for n in range(pred.shape[0]):
            psnr_v += PSNR(pred[n], true[n])
        return psnr_v / pred.shape[0]
        


