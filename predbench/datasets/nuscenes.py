import os.path as osp
from typing import Callable, List, Optional, Sequence, Union, Any
import json
from predbench.registry import DATASETS
from mmengine.dataset import BaseDataset

@DATASETS.register_module()
class NuScenesDataset(BaseDataset):
    def __init__(
        self, 
        clip_len=20,
        frame_interval=1,  
        stride=20,
        
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
        with open(self.ann_file, 'r') as fp:
            date2images = json.load(fp)
        total_frames = self.clip_len * self.frame_interval
        for date, images in date2images.items():
            for start_idx in range(0, len(images)-total_frames, self.stride):
                file_paths = [
                    osp.join(self.data_root, images[start_idx+idx]) for idx in range(total_frames)
                ]
                data = dict(
                    file_paths = file_paths,
                    total_frames = total_frames,
                )
                data_list.append(data)
        print(len(data_list))
        return data_list


