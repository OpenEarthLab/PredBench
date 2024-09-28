from typing import Callable, List, Optional, Sequence, Union, Any
import numpy as np
from predbench.registry import DATASETS
from mmengine.dataset import BaseDataset
import numpy as np
import einops



@DATASETS.register_module()
class TaxiBJDataset(BaseDataset):
    def __init__(self, 
        clip_len=20,
        frame_interval=1,  
        
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
        super().__init__(ann_file, metainfo, data_root, data_prefix, filter_cfg, indices, serialize_data, pipeline, test_mode, lazy_init, max_refetch)


    def load_data_list(self):
        """Load annotation file to get video information."""
        data_list = []
        arrays = np.load(self.data_root)  # data_root = ../taxibj/dataset.npz
        arrays = einops.rearrange(arrays, 'n t c h w -> n t h w c')
        total_frames = self.clip_len * self.frame_interval
        assert total_frames <= 8
        for idx in range(arrays.shape[0]):
            data_info = dict(
                array=arrays[idx],
                total_frames=total_frames,
            )
            data_list.append(data_info)
        
        return data_list
