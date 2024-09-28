import os.path as osp
from typing import Callable, List, Optional, Sequence, Union, Any

import os
import numpy as np
from predbench.registry import DATASETS
from mmengine.dataset import BaseDataset



@DATASETS.register_module()
class CaltechPedestrianDataset(BaseDataset):
    def __init__(
        self, 
        clip_len=20,
        frame_interval=1,  
        stride=1,
        
        ann_file: str  = '', 
        metainfo: dict  = None, 
        data_root: str  = '', 
        data_prefix: dict = dict(img_path=''), 
        filter_cfg: dict  = None, 
        indices: Sequence[int]  = None, 
        serialize_data: bool = True, 
        pipeline: List[Callable[..., Any]] = ..., 
        test_mode: bool = False, 
        lazy_init: bool = False, 
        max_refetch: int = 1000
    ):
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.stride = stride
        super().__init__(ann_file, metainfo, data_root, data_prefix, filter_cfg, indices, serialize_data, pipeline, test_mode, lazy_init, max_refetch)


    def load_data_list(self) -> List[dict]:
        data_list = []
        videos_path = os.listdir(self.data_root)
        total_frames = self.clip_len * self.frame_interval
        for video_path in videos_path:
            video_data = np.load(osp.join(self.data_root, video_path))
            for idx in range(0, video_data.shape[0] - total_frames, self.stride):
                data = dict(
                    array = video_data[idx: idx+total_frames],
                    total_frames = total_frames,
                )
                data_list.append(data)
        print(len(data_list))
        return data_list


