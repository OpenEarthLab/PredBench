import os.path as osp
from typing import Callable, List, Optional, Sequence, Union, Any
import xarray as xr
import einops

import numpy as np
from predbench.registry import DATASETS
from mmengine.dataset import BaseDataset


def fold(data, size=36, stride=12):
    # inverse of unfold/sliding window operation
    # only applicable to the case where the size of the sliding windows is n*stride
    # data (N, size, *)
    # outdata (N_, *)
    # N/size is the number/width of sliding blocks
    assert size % stride == 0
    times = size // stride
    remain = (data.shape[0] - 1) % times
    if remain > 0:
        ls = list(data[::times]) + [data[-1, -(remain * stride):]]
        outdata = np.concatenate(ls, axis=0)  # (36*(151//3+1)+remain*stride, *, 15)
    else:
        outdata = np.concatenate(data[::times], axis=0)  # (36*(151/3+1), *, 15)
    assert outdata.shape[0] == size * ((data.shape[0] - 1) // times + 1) + remain * stride
    return outdata

def data_transform(data, num_years_per_model):
    # data (N, 36, *)
    # num_years_per_model: 151/140
    length = data.shape[0]
    assert length % num_years_per_model == 0
    num_models = length // num_years_per_model
    outdata = np.stack(np.split(data, length / num_years_per_model, axis=0), axis=-1)  # (151, 36, *, 15)
    # cmip6sst outdata.shape = (151, 36, 24, 48, 15) = (year, month, lat, lon, model)
    # cmip5sst outdata.shape = (140, 36, 24, 48, 17)
    # cmip6nino outdata.shape = (151, 36, 15)
    # cmip5nino outdata.shape = (140, 36, 17)
    outdata = fold(outdata, size=36, stride=12) # it is make sense. See official interpretation: https://tianchi.aliyun.com/dataset/98942
    # cmip6sst outdata.shape = (1836, 24, 48, 15), 1836 == 151 * 12 + 24
    # cmip5sst outdata.shape = (1704, 24, 48, 17)
    # cmip6nino outdata.shape = (1836, 15)
    # cmip5nino outdata.shape = (1704, 17)

    # check output data
    assert outdata.shape[-1] == num_models
    assert not np.any(np.isnan(outdata))
    return outdata


def read_raw_data(ds_dir):
    # read and process raw cmip data from CMIP_train.nc and CMIP_label.nc
    # train_cmip = xr.open_dataset(Path(ds_dir) / 'CMIP_train.nc').transpose('year', 'month', 'lat', 'lon')
    # label_cmip = xr.open_dataset(Path(ds_dir) / 'CMIP_label.nc').transpose('year', 'month')
    train_cmip = xr.open_dataset(osp.join(ds_dir, 'CMIP_train.nc')).transpose('year', 'month', 'lat', 'lon')
    label_cmip = xr.open_dataset(osp.join(ds_dir, 'CMIP_label.nc')).transpose('year', 'month')
    # train_cmip.sst.values.shape = (4645, 36, 24, 48)

    # select longitudes
    lon = train_cmip.lon.values
    lon = lon[np.logical_and(lon >= 95, lon <= 330)]
    train_cmip = train_cmip.sel(lon=lon)
    cmip6sst = data_transform(data=train_cmip.sst.values[:2265],
                              num_years_per_model=151)
    cmip5sst = data_transform(data=train_cmip.sst.values[2265:],
                              num_years_per_model=140)
    cmip6nino = data_transform(data=label_cmip.nino.values[:2265],
                               num_years_per_model=151)
    cmip5nino = data_transform(data=label_cmip.nino.values[2265:],
                               num_years_per_model=140)
    # cmip6sst.shape = (1836, 24, 48, 15), cmip6nino.shape = (1836, 15)
    # cmip5sst.shape = (1704, 24, 48, 17), cmip5nino.shape = (1836, 17)
    # 1836=151*12*24 or 1704=140*12+24 represent months' number
    # 15 or 17 represent the number of historical stimulation data by CMIP6 mode
    # 24 and 48 represent latitude and longitude, respectively
    assert len(cmip6sst.shape) == 4
    assert len(cmip5sst.shape) == 4
    assert len(cmip6nino.shape) == 2
    assert len(cmip5nino.shape) == 2
    # store processed data for faster data access
    train_cmip.close()
    label_cmip.close()
    return cmip6sst, cmip5sst, cmip6nino, cmip5nino


def get_data(sst_data, total_frames):
    def prepare_inputs_targets(len_time, total_frames):
        ind = np.arange(0, total_frames).reshape(1, total_frames)
        max_n_sample = len_time - total_frames + 1
        ind = ind + np.arange(max_n_sample)[:, np.newaxis] @ np.ones((1, total_frames), dtype=int)
        return ind

    def cat_over_last_dim(data):
        r"""
        treat different models (15 from CMIP6, 17 from CMIP5) as batch_size
        e.g., cmip6sst.shape = (178, 38, 24, 48, 15), converted_cmip6sst.shape = (2670, 38, 24, 48)
        e.g., cmip5sst.shape = (165, 38, 24, 48, 15), converted_cmip6sst.shape = (2475, 38, 24, 48)
        """
        return np.concatenate(np.moveaxis(data, -1, 0), axis=0)
    
    idx_sst = prepare_inputs_targets(sst_data.shape[0], total_frames)
    sst = cat_over_last_dim(sst_data[idx_sst])
    return sst


@DATASETS.register_module()
class ENSODataset(BaseDataset):
    def __init__(
        self, 
        clip_len=20,
        frame_interval=1,  
        mode='train',
        
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
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        
        super().__init__(ann_file, metainfo, data_root, data_prefix, filter_cfg, indices, serialize_data, pipeline, test_mode, lazy_init, max_refetch)


    def load_data_list(self) -> List[dict]:
        total_frames = self.clip_len*self.frame_interval
        cmip6sst, cmip5sst, cmip6nino, cmip5nino = read_raw_data(self.data_root)
        '''
        Actually, nino index data is not used. 
        This is because it is the sliding average over sst, where window size is 3.
        '''
        if self.mode == 'train':
            sst = np.concatenate(
                [
                    get_data(cmip6sst, total_frames), get_data(cmip5sst[..., :-2], total_frames)
                ], axis=0
            )
        elif self.mode == 'val':
            sst = get_data(cmip5sst[..., -2:-1], total_frames)
        elif self.mode == 'test':
            sst = get_data(cmip5sst[..., -1:], total_frames)
        else:
            raise NotImplementedError
        sst = einops.rearrange(sst, 'n t lat lon -> n t lat lon 1') # n t h w -> n t h w c
        
        data_list = []
        for idx in range(sst.shape[0]):
            data = dict(
                array = sst[idx],
                total_frames = total_frames,
            )
            data_list.append(data)
        return data_list


