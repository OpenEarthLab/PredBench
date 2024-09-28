from typing import Callable, List, Optional, Sequence, Union, Any
from predbench.registry import DATASETS
from mmengine.dataset import BaseDataset


@DATASETS.register_module()
class HumanDataset(BaseDataset):
    def __init__(
        self, 
        clip_len=20,
        frame_interval=1,  
        
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
        super().__init__(ann_file, metainfo, data_root, data_prefix, filter_cfg, indices, serialize_data, pipeline, test_mode, lazy_init, max_refetch)


    def load_data_list(self) -> List[dict]:
        data_list = []
        with open(self.ann_file, 'r') as f:
            annotations = f.readlines()
        total_frames = self.clip_len * self.frame_interval
        for item in annotations:
            filename_prefix, start, end = item.split(',')
            if int(end) - int(start) < total_frames:
                raise RuntimeError
            data = dict(
                frame_dir=self.data_root,
                offset=int(start),
                total_frames=total_frames,  # in fact, total_frames=int(end)-int(start).
                filename_prefix=filename_prefix,
                filename_tmpl='{:0>6d}.jpg',
            )
            data_list.append(data)
        print(len(data_list))
        return data_list


