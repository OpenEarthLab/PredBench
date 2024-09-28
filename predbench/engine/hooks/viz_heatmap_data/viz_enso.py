import cartopy
# for offline usage, maps can be downloaded at https://www.naturalearthdata.com/
cartopy.config['data_dir'] = 'pre_download/cartopy' 

from typing import List
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfe
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from configs._base_.datasets.enso import SST_MAX, SST_MIN


lon = np.arange(95, 330+1, 5)
lat = np.arange(-55, 60+1, 5)
central_longitude = 180
proj = ccrs.PlateCarree(central_longitude=central_longitude)
extend=[95-central_longitude, 330-central_longitude, -55, 60, ]



def visualize_ENSO(
    input: np.array, target: np.array, pred_list: List[np.array], 
    pred_label_list: List[str], save_path: str, plot_strides=[2, 2], font_size=20
):
    '''
    ssts: [t, lat, lon]
    '''
    if pred_label_list != []:
        assert len(pred_list) == len(pred_label_list)
    else: 
        assert len(pred_list) == 1
    min_v = np.min([np.min(input), np.min(target), np.min(pred_list)])
    max_v = np.max([np.max(input), np.max(target), np.max(pred_list)])
    levels = np.arange(min_v, max_v, (max_v - min_v) / 100) # 设置颜色分辨度
    n_cols = np.max([
        input.shape[0] // plot_strides[0], target.shape[0] // plot_strides[1]
    ]) 
    n_rows = 2 + len(pred_list)   # input + output + preds
    fig = plt.figure(dpi=300, figsize=(6*n_cols, 3*n_rows))
    for row, ssts in enumerate([input, target, *pred_list]):
        plot_stride = plot_strides[0] if row == 0 else plot_strides[1]
        for col in range(ssts.shape[0] // plot_stride):
            ax = fig.add_subplot(n_rows, n_cols, row*n_cols+col+1, projection=proj)
            sst = ssts[col*plot_stride]
            
            ax.set_extent(extend, crs=proj)   # 设置经纬度范围
            ax.add_feature(cfe.OCEAN, facecolor='never', linewidth=0.8) # 添加要素
            ax.add_feature(cfe.LAND, edgecolor='black', facecolor='never', linewidth=0.8)
            ax.add_feature(cfe.LAKES, edgecolor='black', facecolor='never', linewidth=0.8)
            ax.add_feature(cfe.RIVERS, edgecolor='black', linewidth=0.8)
            ax.add_feature(cfe.COASTLINE, edgecolor='black', linewidth=0.8)
            cp = ax.contourf(lon-central_longitude, lat, sst, cmap='Spectral_r', levels=levels)
            ax.set_xticks(np.arange(95-central_longitude, 330-central_longitude+1, 47), crs=proj)
            ax.set_yticks(np.arange(-55, 60+1, 23), crs=proj)
            ax.xaxis.set_major_formatter(LongitudeFormatter())
            ax.yaxis.set_major_formatter(LatitudeFormatter())   
    
    n_cols_input = input.shape[0] // plot_strides[0]
    n_cols_target = target.shape[0] // plot_strides[1]   
    fig.axes[0].set_ylabel('input', fontsize=font_size)
    fig.axes[1*n_cols_input].set_ylabel('target', fontsize=font_size)
    for i in range(len(pred_list)):
        if pred_label_list != []:
            fig.axes[n_cols_input+(1+i)*n_cols_target].set_ylabel(
                f'{pred_label_list[i]}\nprediction', fontsize=font_size
            ) 
        else: 
            fig.axes[n_cols_input+(1+i)*n_cols_target].set_ylabel(
                'prediction', fontsize=font_size
            ) 
    fig.tight_layout()
    plt.colorbar(cp, ax=fig.axes)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()