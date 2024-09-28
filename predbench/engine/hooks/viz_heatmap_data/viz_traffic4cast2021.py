import numpy as np
import einops
import matplotlib.pyplot as plt
from typing import List



def visualize_Traffic4cast2021(
    input: np.array, target: np.array, pred_list: List[np.array], 
    pred_label_list: List[str], save_path: str, plot_strides=[1,1], font_size=20
):
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
    fig = plt.figure(dpi=300, figsize=(3*n_cols+2, 3*n_rows))
    for row, ssts in enumerate([input, target, *pred_list]):
        plot_stride = plot_strides[0] if row == 0 else plot_strides[1]
        for col in range(ssts.shape[0] // plot_stride):
            ax = fig.add_subplot(n_rows, n_cols, row*n_cols+col+1)
            sst = ssts[col*plot_stride]
            cp = ax.contourf(sst, cmap='PuBu', levels=levels)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.yaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_formatter(plt.NullFormatter())
    
    n_cols_input = input.shape[0] // plot_strides[0]
    n_cols_target = target.shape[0] // plot_strides[1]    
    fig.axes[0].set_ylabel('input', fontsize=font_size)
    fig.axes[1*n_cols].set_ylabel('target', fontsize=font_size)
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
    


