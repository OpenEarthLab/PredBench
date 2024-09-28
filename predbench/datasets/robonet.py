import os.path as osp
from typing import Callable, List, Optional, Sequence, Union, Any
import random
import pandas as pd
from mmengine.fileio import load

from predbench.registry import DATASETS
from mmengine.dataset import BaseDataset



@DATASETS.register_module()
class RoboNetDataset(BaseDataset):
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
        meta_data = pd.read_pickle(osp.join(self.data_root, 'meta_data.pkl'), compression='gzip')
        meta_data = meta_data.loc[: , ['sha256', 'img_T', 'ncam', 'image_format', 'img_encoding']]
        '''
        all_columns_in_meta_data = [
            'action_T', 'action_space', 'adim', 'background', 'bin_insert',
            'bin_type', 'camera_configuration', 'camera_type',
            'contains_annotation', 'environment_size', 'file_version', 'frame_dim',
            'gripper', 'high_bound', 'image_format', 'img_T', 'img_encoding',
            'low_bound', 'ncam', 'object_batch', 'object_classes', 'policy_desc',
            'primitives', 'robot', 'sdim', 'sha256', 'state_T', 'term_t', 'traj_ok'
        ]
        '''
        annotations = load(self.ann_file)
        for idx in annotations:
            hdf5_data = osp.join(self.data_root, idx)
            # actually, all the paths are existent !!!
            # if not (os.path.exists(hdf5_data) and os.path.isfile(hdf5_data)):
            #     print(f'{hdf5_data} does not exist!!!')
            #     continue    # only reserve the existent hdf5 file
            meta_data_idx = meta_data.loc[idx]
            if meta_data_idx['img_T'] < self.clip_len*self.frame_interval:
                continue
            if self.test_mode:
                cam_to_load = random.choice(range(meta_data_idx['ncam']))
            else:
                cam_to_load = 0
            data_info = dict(
                hdf5_data = hdf5_data,
                meta_data = meta_data_idx,
                total_frames = meta_data_idx['img_T'],
                cam_to_load = cam_to_load,
            )
            data_list.append(data_info)
        print(len(data_list))
        return data_list   
            
