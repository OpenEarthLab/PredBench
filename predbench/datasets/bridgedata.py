import copy
import os.path as osp
from typing import List, Sequence, Callable, Any
import numpy as np
from predbench.registry import DATASETS
from mmengine.dataset import BaseDataset
import numpy as np


from mmengine.dataset import BaseDataset
from mmengine.fileio import load

@DATASETS.register_module()
class BridgeDataDataset(BaseDataset):
    def __init__(
        self, 
        clip_len=20,
        frame_interval=1,  
        stride=None,
        # mode='train',
        
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
        # self.mode = mode
        super().__init__(ann_file, metainfo, data_root, data_prefix, filter_cfg, indices, serialize_data, pipeline, test_mode, lazy_init, max_refetch)


    def load_data_list(self) -> List[dict]:
        data_list = []
        images = load(self.ann_file)
        for image, images_len in images.items():
            if images_len < self.clip_len * self.frame_interval:
                continue
            if self.stride != None:
                for offset in range(0, images_len - self.clip_len*self.frame_interval+1, self.stride):
                    data = dict(
                        offset=offset,
                        frame_dir=osp.join(self.data_root, image),
                        total_frames=self.clip_len*self.frame_interval,  
                        filename_prefix='im_',
                        filename_tmpl='{}.jpg',
                    )
                    data_list.append(data)
            else:
                data = dict(
                    frame_dir=osp.join(self.data_root, image),
                    total_frames=images_len,  
                    filename_prefix='im_',
                    filename_tmpl='{}.jpg',
                )
                data_list.append(data)
        print(len(data_list))
        return data_list
    
    