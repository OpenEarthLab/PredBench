from typing import Callable, List, Optional, Sequence, Union, Any
import numpy as np
from predbench.registry import DATASETS
from mmengine.dataset import BaseDataset


@DATASETS.register_module()
class MovingMNISTDataset(BaseDataset):
    def __init__(
        self,
        fixed_data=None,
        clip_len=20,
        frame_interval=1,
        n_epoch=200,
        
        ann_file: str  = '', 
        metainfo: dict  = None, 
        data_root: str  = '', 
        data_prefix: dict = dict(img_path=''), 
        filter_cfg: dict  = None, 
        indices: Sequence[int]  = None, 
        serialize_data: bool = True, 
        pipeline: List[dict] = ..., 
        test_mode: bool = False, 
        lazy_init: bool = False, 
        max_refetch: int = 1000,
    ):
        self.fixed_data = fixed_data
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.n_epoch = n_epoch
        super().__init__(ann_file, metainfo, data_root, data_prefix, filter_cfg, indices, serialize_data, pipeline, test_mode, lazy_init, max_refetch)
    
    def load_data_list(self):
        """Load annotation file to get video information."""
        data_list = []
        total_frames = int(self.clip_len*self.frame_interval)
        if self.test_mode:
            video_data = np.load(self.fixed_data)
            print(self.fixed_data)
            print(video_data.shape)
            
            for idx in range(video_data.shape[0]):
                video_info = dict(
                    array = video_data[idx],
                    total_frames = total_frames,
                )
                data_list.append(video_info)
        else:
            assert isinstance(self.fixed_data, int), 'generate the training dataset of fixed number on the fly'
            video_info = dict(
                total_frames = total_frames,
            )
            data_list = [video_info] * self.fixed_data
            
        return data_list
    
