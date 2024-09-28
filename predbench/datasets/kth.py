import os.path as osp
from typing import Callable, List, Optional, Sequence, Union, Any
from PIL import Image
import einops
import os
import numpy as np
from predbench.registry import DATASETS
from mmengine.dataset import BaseDataset


@DATASETS.register_module()
class KTHDataset(BaseDataset):
    def __init__(
        self, 
        clip_len=20,
        frame_interval=1, 
        stride=20,
        mode='train', 
        
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
        self.mode = mode
        super().__init__(ann_file, metainfo, data_root, data_prefix, filter_cfg, indices, serialize_data, pipeline, test_mode, lazy_init, max_refetch)


    def load_data_list(self) -> List[dict]:
        total_frames = self.clip_len * self.frame_interval
        print(total_frames, self.clip_len, self.frame_interval)
        data_processor = DataProcess(seq_length=total_frames, stride=self.stride)
        dataset, indices = data_processor.load_data(self.data_root, mode=self.mode)
        print(dataset.shape)
        data_list = []
        for idx in indices:
            data = dict(
                array = dataset[idx: idx + total_frames],
                total_frames = total_frames
            )
            data_list.append(data)
        
        return data_list




class DataProcess(object):
    """Class for preprocessing dataset inputs."""

    def __init__(self, seq_length, stride):
        self.category_1 = ['boxing', 'handclapping', 'handwaving', 'walking']
        self.category_2 = ['jogging', 'running']
        self.category = self.category_1 + self.category_2

        self.train_person = [
            '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
            '13', '14',
        ]
        self.val_person = ['15', '16']
        self.test_person = ['17', '18', '19', '20', '21', '22', '23', '24', '25']

        self.seq_len = seq_length
        self.stride = stride

    def load_data(self, path, mode='train'):
        """Loads the dataset.
        Args:
            path: action_path.
            mode: Training or testing.
        Returns:
            A dataset and indices of the sequence.
        """
        data_path = osp.join(path, f'{mode}_data.npy')
        indices_path = osp.join(path, f'{mode}_indices.npy')
        if osp.exists(data_path) and osp.exists(indices_path):  # use cache
            data = np.load(data_path).astype('float')
            indices = np.load(indices_path)
            print(f'using saved cache, {mode} mode')
            print('there are ' + str(data.shape[0]) + ' pictures')
            print('there are ' + str(len(indices)) + ' sequences')
            return data, indices
    
        # path = paths[0]
        if mode == 'train':
            person_id = self.train_person
        elif mode == 'val':
            person_id = self.val_person
        elif mode.startswith('test'):   # test, test_2x_pred_len, test_extrapolation, test_generalization, test_robustness
            person_id = self.test_person
        else:
            print('begin load data' + str(path))

        frames_np = []
        frames_file_name = []
        frames_person_mark = []
        frames_category = []
        person_mark = 0
        
        c_dir_list = self.category
        frame_category_flag = -1
        for c_dir in c_dir_list:  # handwaving
            if c_dir in self.category_1:
                frame_category_flag = 1  # 20 step
            elif c_dir in self.category_2:
                frame_category_flag = 2  # 3 step
            else:
                print('category error!!!')

            c_dir_path = os.path.join(path, c_dir)
            p_c_dir_list = os.listdir(c_dir_path)
            # p_c_dir_list.sort() # for date seq

            for p_c_dir in p_c_dir_list:  # person01_handwaving_d1_uncomp
                # print(p_c_dir)
                if p_c_dir[6:8] not in person_id:
                    continue
                person_mark += 1

                dir_path = os.path.join(c_dir_path, p_c_dir)
                filelist = os.listdir(dir_path)
                filelist.sort()  # tocheck
                for cur_file in filelist:  # image_0257
                    if not cur_file.startswith('image'):
                        continue

                    frame_im = Image.open(os.path.join(dir_path, cur_file))
                    frame_np = np.array(frame_im)  # (1000, 1000) numpy array
                    # print(frame_np.shape)
                    frame_np = frame_np[:, :, 0]  #
                    frames_np.append(frame_np)
                    frames_file_name.append(cur_file)
                    frames_person_mark.append(person_mark)
                    frames_category.append(frame_category_flag)

        # is it a begin index of sequence
        indices = []
        index = len(frames_person_mark) - 1
        while index >= self.seq_len - 1:
            if frames_person_mark[index] == frames_person_mark[index - self.seq_len + 1]:
                end = int(frames_file_name[index][6:10])
                start = int(frames_file_name[index - self.seq_len + 1][6:10])
                # TODO(yunbo): mode == 'test'
                if end - start == self.seq_len - 1:
                    indices.append(index - self.seq_len + 1)
                    if frames_category[index] == 1:
                        index -= self.stride - 1
                    elif frames_category[index] == 2:
                        index -= 2
                    else:
                        print('category error 2 !!!')
            index -= 1

        frames_np = np.asarray(frames_np)
        data = einops.rearrange(frames_np, 'n h w -> n h w 1')
        print('there are ' + str(data.shape[0]) + ' pictures')
        print('there are ' + str(len(indices)) + ' sequences')
        
        np.save(osp.join(path, f'{mode}_data.npy'), data.astype('uint8'))
        np.save(osp.join(path, f'{mode}_indices.npy'), np.asarray(indices).astype('int32'))
        print(f'save cache in {mode}_data.npy, {mode}_indices.npy')
        return data, indices

    