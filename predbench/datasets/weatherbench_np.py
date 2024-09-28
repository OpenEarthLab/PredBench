import os.path as osp
from typing import Callable, List, Optional, Sequence, Union, Any

import os
from predbench.registry import DATASETS

from mmengine.dataset import BaseDataset

@DATASETS.register_module()
class WeatherBenchDatasetNp(BaseDataset):
    def __init__(
        self, 
        clip_len=20,
        frame_interval=1,  
        year_split=[],
        hour_stride=24,
        
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
        self.start_year = int(year_split[0])
        self.end_year = int(year_split[1])
        self.stride = hour_stride
        super().__init__(ann_file, metainfo, data_root, data_prefix, filter_cfg, indices, serialize_data, pipeline, test_mode, lazy_init, max_refetch)


    def load_data_list(self) -> List[dict]:
        data_list = []
        total_frames = self.clip_len * self.frame_interval
        all_years = [dir for dir in os.listdir(self.data_root) if osp.isdir(osp.join(self.data_root, dir))]
        for year in all_years:
            if self.start_year <= int(year) <= self.end_year:
                npy_dir = osp.join(self.data_root, year)
                for idx in range(0, len(os.listdir(npy_dir))-total_frames, self.stride):
                    data = dict(
                        npy_dir = npy_dir,
                        offset = idx,
                        year = year,
                        total_frames = total_frames,
                        filename_prefix = year,
                        filename_tmpl = '-{:0>4d}.npy',
                    )
                    data_list.append(data)
        print(len(data_list))
        return data_list


