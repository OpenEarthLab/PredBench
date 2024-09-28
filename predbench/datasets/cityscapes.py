from typing import Callable, List, Optional, Sequence, Union, Any
import numpy as np
from predbench.registry import DATASETS
from mmengine.dataset import BaseDataset
import h5py
import numpy as np


# CityScapes dataset and kitti dataset are alomst the same
@DATASETS.register_module()
class CityScapesDataset(BaseDataset):
    def __init__(self, 
        clip_len=20,
        frame_interval=1,  
        hdf5_data='',      
        stride=10,
        mode='',
        
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
        self.hdf5_data = hdf5_data
        self.stride = stride
        self.mode = mode
        super().__init__(ann_file, metainfo, data_root, data_prefix, filter_cfg, indices, serialize_data, pipeline, test_mode, lazy_init, max_refetch)


    def load_data_list(self):
        """Load annotation file to get video information."""
        data_list = []
        total_frames = self.clip_len * self.frame_interval
        with h5py.File(self.hdf5_data, 'r') as hdf5_data:
            for k in hdf5_data['len'].keys():
                video_hdf5 = hdf5_data[k]
                video = []
                for idx in range(len(video_hdf5)):
                    video.append(
                        np.array(video_hdf5[str(idx)])
                    )
                video = np.stack(video, axis=0)
                if self.mode == 'test':
                    indices = [0]
                else:
                    indices = range(0, video.shape[0]-total_frames+1, self.stride)
                for idx in indices:
                    data_info = dict(
                            array = video[idx: idx+total_frames],
                            total_frames = total_frames,
                        )
                    data_list.append(data_info)
        print(len(data_list))
        return data_list
