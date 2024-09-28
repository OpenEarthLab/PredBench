from typing import Optional, Sequence, Union
from predbench.registry import MODELS
from mmengine.model import BaseDataPreprocessor, ImgDataPreprocessor
import torch

@MODELS.register_module()
class VideoProcessor(BaseDataPreprocessor):
    def __init__(
        self, 
        input_len,
        output_len,
        pred_len,
        mean: Optional[Sequence[Union[float, int]]] = None,
        std: Optional[Sequence[Union[float, int]]] = None,
        non_blocking: bool = False
    ):
        super().__init__(non_blocking)
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.pred_len = int(pred_len)
        assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def forward(self, data, training: bool = False, post_process=False):
        C = len(self.mean)
        if post_process == False:   # pre-process, normalize data
            data = self.cast_data(data)
            batch_imgs = torch.stack(data['imgs'], dim=0).to(torch.float32)
            assert len(batch_imgs.shape) == 5 # b t c h w
            # print(batch_imgs.shape)
            assert batch_imgs.shape[-3] == C
            # print(batch_imgs.dtype)
            # print(self.mean, self.std)
            # print(batch_imgs.shape)
            # print('before pre-process', torch.max(batch_imgs), torch.min(batch_imgs))
            for i in range(C):
                batch_imgs[:,:,i,...] = (batch_imgs[:,:,i,...] - self.mean[i]) / self.std[i]
            # print('after pre-process', torch.max(batch_imgs), torch.min(batch_imgs))
            # print(batch_imgs.dtype)
            if training:    # train, input_len->output_len
                data = dict(
                    input = batch_imgs[:, 0: self.input_len, ...],
                    target = batch_imgs[:, self.input_len: self.input_len+self.output_len, ...],
                )
            else:   # val or test, input_len->pred_len
                data = dict(
                    input = batch_imgs[:, 0: self.input_len, ...],
                    target = batch_imgs[:, self.input_len: self.input_len+self.pred_len, ...],
                )
            return data
        else:   # post-process, denormalize data
            pred = data['pred']
            gt = data['gt']
            # print('before post-process', torch.max(pred), torch.min(pred), torch.max(gt), torch.min(gt))
            # print(self.std, self.mean)
            for i in range(C):
                pred[:,:,i,...] = pred[:,:,i,...] * self.std[i] + self.mean[i]
                gt[:,:,i,...] = gt[:,:,i,...] * self.std[i] + self.mean[i]
            # print('after post-process', torch.max(pred), torch.min(pred), torch.max(gt), torch.min(gt))
            return dict(pred=pred, gt=gt)