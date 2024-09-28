
# dataset settings
_base_ = [
    '../../_base_/datasets/weatherbench128x256_69.py', 
    '../../_base_/default_runtime.py', 
    '../../_base_/models/mcvd.py',
    '../../_base_/schedules/schedule_1e6i.py',
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', 
        by_epoch=False, interval=int(1e4), max_keep_ckpts=1, save_best='error/mse', rule='less'
    ),
)

custom_hooks = [dict(type='VizVideoHook', data_type='weatherbench', viz_stages=['val','test']), dict(type='EMAHook', momentum=0.001)]

val_evaluator=[
    dict(type='ErrorMetrics', metric_list=['mae', 'mse', 'rmse'], norm_01=False),
]
test_evaluator=[
    dict(type='ErrorMetrics', metric_list=['mae', 'mse', 'rmse'], norm_01=False),
    dict(type='WeatherMetrics', metric_list=['wmae', 'wmse', 'wrmse', 'acc'], latitude=_base_.latitude, metric_channels=_base_.metric_channels),
]


data_processor=dict(
    type='VideoProcessor',
    input_len=_base_.input_len,
    output_len=_base_.output_len,
    pred_len=_base_.pred_len,
    mean=_base_.img_norm_cfg['mean'],
    std=_base_.img_norm_cfg['std'],
)
