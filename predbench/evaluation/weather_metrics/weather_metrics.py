from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
from mmengine.evaluator import BaseMetric
import einops

from predbench.registry import METRICS
from .enso import sst_to_nino, compute_enso_score


@METRICS.register_module()
class WeatherMetrics(BaseMetric):
    def __init__(
        self, 
        metric_list: Union[str, Tuple[str]] = ['log_bias', 'csi', 'pod', 'sucr'],
        threshold_list: Optional[Tuple[float]] = None,
        latitude: Optional[List] = None,
        metric_channels: Optional[List] = None,
        by_frame: bool = False,
        eps: float = 1e-4,
        collect_device: str = 'cpu', 
        prefix: str  = 'weather', 
        collect_dir: str  = None
    ) -> None:
        super().__init__(collect_device, prefix, collect_dir)
        self.basic_metrics_fn = dict(
            wmae=self.cal_WeightedMAE, wmse=self.cal_WeightedMSE,
            wrmse=self.cal_WeightedRMSE, acc=self.cal_ACC,
        )
        self.categorical_metrics_fn = dict(
            csi=self.cal_CSI,  ets=self.cal_ETS, far=self.cal_FAR, mar=self.cal_MAR, 
            pod=self.cal_POD, sucr=self.cal_SUCR, bias=self.cal_BIAS, 
            log_bias=self.cal_LogBIAS, hss=self.cal_HSS,
        )
        self.extra_metrics_fn = {'nino3.4': sst_to_nino,}
        
        self.basic_metric_list = [
            metric for metric in metric_list if metric in self.basic_metrics_fn
        ]
        self.categorical_metric_list = [
            metric for metric in metric_list if metric in self.categorical_metrics_fn
        ]
        self.extra_metric_list = [
            metric for metric in metric_list if metric in self.extra_metrics_fn
        ]
        
        self.threshold_list = threshold_list
        self.latitude = latitude
        self.metric_channels = metric_channels
        self.by_frame = by_frame
        self.eps = eps
        

        
    
    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        pred, gt = data_samples[0]['pred'], data_samples[0]['gt']   # data in original space
        result = dict() 
        for metric in self.basic_metric_list:
            if self.metric_channels == None or self.metric_channels == []:
                pred_m, gt_m = pred, gt
            else:
                pred_m, gt_m = pred[:,:,self.metric_channels], gt[:,:,self.metric_channels] 
            clims = torch.stack(data_batch['clims'], dim=0)[:,-pred.shape[1]:].to(pred)
            if not self.by_frame:
                result[metric] = self.basic_metrics_fn[metric](
                    pred_m, gt_m, lat=self.latitude, clims=clims
                )
            else:
                assert len(pred.shape) == 5, f'image shape is {pred.shape}, only video is supported'
                metric_res = []
                for t in range(pred.shape[1]):
                    metric_res.append(self.basic_metrics_fn[metric](
                        pred_m[:,t], gt_m[:,t], lat=self.latitude, clims=clims[:, t]
                    ))
                result[metric] = torch.stack(metric_res)
        for metric in self.extra_metric_list:
            result[metric] = self.extra_metrics_fn[metric](pred, gt)

        if len(self.categorical_metric_list) != 0:
            t_hits = list()
            t_misses = list()
            t_fas = list()
            t_crs = list()
            for threshold in self.threshold_list:
                # print('pred', torch.max(pred), torch.min(pred))
                # print('gt', torch.max(gt), torch.min(gt))
                hits, misses, fas, crs = self.get_contingency_table(pred, gt, threshold)
                t_hits.append(hits)
                t_misses.append(misses)
                t_fas.append(fas)
                t_crs.append(crs)
            result.update(dict(hits=t_hits, misses=t_misses, fas=t_fas, crs=t_crs))
        self.results.append(result)
        
    def compute_metrics(self, results: list) -> dict:
        metrics = dict()
        for metric in self.basic_metric_list:
            results_tensor = torch.stack([result[metric] for result in results], dim=0)
            metrics[metric] = results_tensor.mean(dim=0)
        for metric in self.extra_metric_list:
            if metric == 'nino3.4':
                results_tensor = torch.cat([result[metric] for result in results], dim=1)
                nino_pred, nino_true = results_tensor[0], results_tensor[1]
                # print(torch.max(nino_pred), torch.min(nino_pred))
                # print(torch.max(nino_true), torch.min(nino_true))
                # print(nino_pred.shape, nino_true.shape)
                nino_acc, nino_rmse = compute_enso_score(nino_pred, nino_true, acc_weight=None)
                w_nino_acc, _ = compute_enso_score(nino_pred, nino_true, acc_weight='default')
                metrics['C_nino3.4_m'] = nino_acc
                metrics['C_nino3.4_wm'] = w_nino_acc
                metrics['nino_rmse'] = nino_rmse
        if len(self.categorical_metric_list) != 0:
            categorical_metric_params = dict()
            for k in ['hits', 'misses', 'fas', 'crs']:
                categorical_metric_params[k] = torch.tensor([result[k] for result in results]).sum(dim=0)
            # print(general_metric_params)
            categorical_metric_params.update(dict(eps=self.eps))
            for metric in self.categorical_metric_list:
                score_sum = 0
                scores = self.categorical_metrics_fn[metric](**categorical_metric_params)
                for i, threshold in enumerate(self.threshold_list):
                    score = scores[i].item()
                    metrics[f'{metric}_{threshold}'] = score
                    score_sum += score
                metrics[f'{metric}_avg'] = score_sum / len(self.threshold_list)   
        return metrics

    def _threshold(self, pred, target, T):
        """
        Returns binary tensors t,p the same shape as target & pred.  t = 1 wherever
        target > t.  p =1 wherever pred > t.  p and t are set to 0 wherever EITHER
        t or p are nan.
        This is useful for counts that don't involve correct rejections.

        Parameters
        ----------
        target
            torch.Tensor
        pred
            torch.Tensor
        T
            numeric_type:   threshold
        Returns
        -------
        t
        p
        """
        t = (target >= T).float()
        p = (pred >= T).float()
        is_nan = torch.logical_or(torch.isnan(target),
                                torch.isnan(pred))
        t[is_nan] = 0
        p[is_nan] = 0
        return t, p
    
    @staticmethod
    def cal_WeightedRMSE(pred, gt, lat, clims):
        '''
        weighted root mean square error
        calculate weighted rmse for every channel without spatial_norm
        gt and pred has the shape of 
            [n t c h w] 
        lat represents latitude and corresponds to h, its value must be in (-90, 90)
        '''
        if lat != None and lat != []:
            weights_lat = torch.cos(torch.deg2rad(torch.tensor(lat))).to(pred.device)
            weights_lat /= weights_lat.mean()
        else:
            weights_lat = torch.ones(pred.shape[-2], device=pred.device)
        
        
        
        weights_lat = einops.rearrange(weights_lat, 'h -> 1 1 h 1')
        wrmse = torch.mean(
            torch.sqrt(torch.mean(weights_lat * (pred-gt)**2, dim=(-1,-2))),
            dim=0
        )
        
        return wrmse
    
    @staticmethod
    def cal_WeightedMSE(pred, gt, lat, clims):
        '''
        weighted mean square error
        calculate weighted mse for every channel without spatial_norm
        gt and pred has the shape of 
            [n t c h w] 
        lat represents latitude and corresponds to h, its value must be in (-90, 90)
        '''
        if lat != None and lat != []:
            weights_lat = torch.cos(torch.deg2rad(torch.tensor(lat))).to(pred.device)
            weights_lat /= weights_lat.mean()
        else:
            weights_lat = torch.ones(pred.shape[-2], device=pred.device)
        weights_lat = einops.rearrange(weights_lat, 'h -> 1 1 1 h 1')
        
        weighted_error = weights_lat * (pred-gt)**2
        return torch.mean(weighted_error, dim=(0,1,3,4)).detach().cpu()
        
    
    @staticmethod
    def cal_WeightedMAE(pred, gt, lat, clims):
        '''
        weighted mean absolute error
        calculate weighted mae for every channel without spatial_norm
        gt and pred has the shape of 
            [n t c h w] 
        lat represents latitude and corresponds to h, its value must be in (-90, 90)
        '''
        if lat != None and lat != []:
            weights_lat = torch.cos(torch.deg2rad(torch.tensor(lat))).to(pred.device)
            weights_lat /= weights_lat.mean()
        else:
            weights_lat = torch.ones(pred.shape[-2], device=pred.device)
        weights_lat = einops.rearrange(weights_lat, 'h -> 1 1 1 h 1')
        
        weighted_error = weights_lat * torch.abs(pred-gt)
        return torch.mean(weighted_error, dim=(0,1,3,4)).detach().cpu()
        
    
    @staticmethod
    def cal_ACC(pred, gt, lat, clims):
        '''
        Anomaly Correlation Coefficient
        calculate acc for every channel
        '''
        if lat != None and lat != []:
            weights_lat = torch.cos(torch.deg2rad(torch.tensor(lat))).to(pred.device)
            weights_lat /= weights_lat.mean()
        else:
            weights_lat = torch.ones(pred.shape[-2], device=pred.device)
        weights_lat = einops.rearrange(weights_lat, 'h -> 1 1 h 1')        
        
        pred_prime = pred - clims
        gt_prime = gt - clims
        if len(pred_prime.shape) == 5:
            pred_prime = einops.rearrange(pred_prime, 'n t c h w -> (n t) c h w')
            gt_prime = einops.rearrange(gt_prime, 'n t c h w -> (n t) c h w')
        return (
            torch.sum(weights_lat*gt_prime*pred_prime, dim=(-1,-2)) / 
            torch.sqrt(
                torch.sum(weights_lat*(gt_prime**2), dim=(-1,-2)) * 
                torch.sum(weights_lat*(pred_prime**2), dim=(-1,-2))
            )
        ).mean(dim=0).detach().cpu()
        
        
        

    def get_contingency_table(self, pred, target, threshold):
        """
        Parameters
        ----------
        pred, target:   torch.Tensor
        threshold:  int

        Returns
        -------
        hits, misses, fas (false alarms), crs (correct rejections):  torch.Tensor
            each has shape (seq_len, )

        contingency table
        -------
                        |   Predictive Positive |   Predictive Negative
        ---------------------------------------------------------------------
        True Positive   |           hits        |       misses
        True Negative   |       false alarms    |   correct rejections
        """
        with torch.no_grad():
            t, p = self._threshold(pred, target, threshold)
            hits = torch.sum(t * p).int()
            misses = torch.sum(t * (1 - p)).int()
            fas = torch.sum((1 - t) * p).int()
            crs = torch.sum((1 - t) * (1 - p)).int()
        return hits, misses, fas, crs

    @staticmethod
    def cal_CSI(hits, misses, fas, crs, eps):
        '''
        critical success index
        '''
        return hits / (hits + misses + fas + eps)

    @staticmethod
    def cal_ETS(hits, misses, fas, crs, eps):
        '''
        equitable threat score
        details in the paper:
        Winterrath, T., & Rosenow, W. (2007). A new module for the tracking of
        radar-derived precipitation with model-derived winds.
        Advances in Geosciences,10, 77-83. https://doi.org/10.5194/adgeo-10-77-2007
        '''
        num = (hits + fas) * (hits + misses)
        den = hits + misses + fas + crs
        Dr = num / den
        return (hits - Dr) / (hits + misses + fas - Dr + eps)

    @staticmethod
    def cal_FAR(hits, misses, fas, crs, eps):
        '''
        false alarm rate
        '''
        return fas / (hits + fas + eps)

    @staticmethod
    def cal_MAR(hits, misses, fas, crs, eps):
        '''
        missing alarm rate
        '''
        return misses / (hits + misses + eps)

    @staticmethod
    def cal_POD(hits, misses, fas, crs, eps):
        '''
        probability of detection
        '''
        return hits / (hits + misses + eps)

    @staticmethod
    def cal_SUCR(hits, misses, fas, crs, eps):
        '''
        success rate
        '''
        return hits / (hits + fas + eps)

    @staticmethod
    def cal_BIAS(hits, misses, fas, crs, eps):
        '''
        bias score
        '''
        bias = (hits + fas) / (hits + misses + eps)
        return bias

    @staticmethod
    def cal_LogBIAS(hits, misses, fas, crs, eps):
        '''
        bias score in log scale
        '''
        bias = (hits + fas) / (hits + misses + eps)
        logbias = torch.pow(bias / torch.log(torch.tensor(2.0)), 2.0)
        return logbias

    @staticmethod
    def cal_HSS(hits, misses, fas, crs, eps):
        '''
        Heidke skill score
        '''
        hss_num = 2 * (hits * crs - misses * fas)
        hss_den = misses**2 + fas**2 + 2 * hits * crs + (misses + fas) * (hits + crs)
        return hss_num / (hss_den + eps)
