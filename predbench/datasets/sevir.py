import os.path as osp
from typing import Callable, List, Optional, Sequence, Union, Any
import os
import numpy as np
from predbench.registry import DATASETS
from mmengine.dataset import BaseDataset

SEVIR_FRAME_NUM = 49

@DATASETS.register_module()
class SEVIRDataset(BaseDataset):
    def __init__(
        self, 
        clip_len=20,
        frame_interval=1,  
        frame_stride=12,
        
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
        max_refetch: int = 1000
    ):
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.frame_stride = frame_stride
        super().__init__(ann_file, metainfo, data_root, data_prefix, filter_cfg, indices, serialize_data, pipeline, test_mode, lazy_init, max_refetch)


    def load_data_list(self) -> List[dict]:
        data_list = []
        file_names = os.listdir(self.data_root)
        total_frames = self.clip_len * self.frame_interval
        assert total_frames <= SEVIR_FRAME_NUM
        for file_name in file_names:
            for idx in range(0, SEVIR_FRAME_NUM-total_frames+1, self.frame_stride):
                data = dict(
                    file_name=osp.join(self.data_root, file_name),
                    offset=idx,
                    total_frames=total_frames,
                )
                data_list.append(data)
        return data_list


