import os.path as osp
from typing import Callable, List, Optional, Sequence, Union, Any
from predbench.registry import DATASETS
from mmengine.dataset import BaseDataset
from mmengine.fileio import load



@DATASETS.register_module()
class KittiDataset(BaseDataset):
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
        annotations = load(self.ann_file)
        total_frames = self.clip_len * self.frame_interval
        for frame_dir, frame_num in annotations.items():
            if frame_num < total_frames:
                continue
            for idx in range(0, frame_num - total_frames, self.stride):
                data = dict(
                    frame_dir=osp.join(self.data_root, frame_dir),
                    offset=idx,
                    total_frames=total_frames,
                    filename_tmpl='{}.png',
                )
                data_list.append(data)
        print(len(data_list))
        return data_list


