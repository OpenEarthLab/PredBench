
# dataset settings
_base_ = [
    '../../_base_/datasets/mnist.py', 
    '../../_base_/default_runtime.py', 
    '../../_base_/models/mcvd.py',
    '../../_base_/schedules/schedule_5e5i.py',
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', 
        by_epoch=False, interval=int(1e4), max_keep_ckpts=1, save_best='error/mse', rule='less'
    ),
)

custom_hooks = [dict(type='VizVideoHook', viz_stages=['val','test']), dict(type='EMAHook', momentum=0.001)]

img_norm_cfg = dict(
    mean=[127.5], std=[127.5], to_bgr=False
)

val_evaluator=[
    dict(type='ErrorMetrics', metric_list=['mae', 'mse', 'rmse'], norm_01=False),
]
test_evaluator=[
    dict(type='ErrorMetrics', metric_list=['mae', 'mse', 'rmse'], norm_01=False),
    dict(type='SimilarityMetrics', is_img=True, metric_list=['ssim', 'psnr', 'snr'], norm_01=False),
    dict(type='PerceptionMetrics', metric_list=['lpips', 'fvd'], collect_device='gpu', norm_01=False),
]


data_processor=dict(
    type='VideoProcessor',
    input_len=_base_.input_len,
    output_len=_base_.output_len,
    pred_len=_base_.pred_len,
    mean=img_norm_cfg['mean'],
    std=img_norm_cfg['std'],
)
