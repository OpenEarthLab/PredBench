
# dataset settings
dataset_type = 'ENSODataset'
data_root = 'data/ICARENSO/enso_round1_train_20210201'


NINO_WINDOW_T = 3  # Nino index is the sliding average over sst, window size is 3
CMIP6_SST_MAX = 10.198975563049316
CMIP6_SST_MIN = -16.549121856689453
CMIP5_SST_MAX = 8.991744995117188
CMIP5_SST_MIN = -9.33076286315918
CMIP6_NINO_MAX = 4.138188362121582
CMIP6_NINO_MIN = -3.5832221508026123
CMIP5_NINO_MAX = 3.8253555297851562
CMIP5_NINO_MIN = -2.691682815551758
SST_MAX = max(CMIP6_SST_MAX, CMIP5_SST_MAX)
SST_MIN = min(CMIP6_SST_MIN, CMIP5_SST_MIN)

img_H = 24  # latitude
img_W = 48  # longitude
img_C = 1
img_norm_cfg = dict(
    mean=[SST_MIN], std=[SST_MAX - SST_MIN], to_bgr=False
)

frame_interval=1
input_len=12
output_len=14
pred_len=14

train_frame_interval=1
train_input_len=input_len
train_output_len=output_len
train_clip_len=input_len + output_len


val_frame_interval=frame_interval
val_input_len=input_len
val_output_len=pred_len
val_clip_len=val_input_len + val_output_len

test_frame_interval=frame_interval
test_input_len=input_len
test_output_len=pred_len
test_clip_len=test_input_len + test_output_len

input_shape=(input_len,img_C,img_H,img_W)
batch_size=16
train_pipeline = [
    dict(
        type='SampleFrames', 
        clip_len=train_clip_len, 
        frame_interval=train_frame_interval, 
        num_clips=1,
        test_mode=False,
    ),
    dict(type='ArrayDecode', modality='Other'),
    # dict(type='Resize', scale=(img_W, img_H), keep_ratio=False),
    dict(type='FormatShape', input_format='TCHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=val_clip_len,
        frame_interval=val_frame_interval,
        num_clips=1,
        test_mode=True),
    dict(type='ArrayDecode', modality='Other'),
    # dict(type='Resize', scale=(img_W, img_H), keep_ratio=False),
    dict(type='FormatShape', input_format='TCHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=test_clip_len,
        frame_interval=test_frame_interval,
        num_clips=1,
        test_mode=True),
    dict(type='ArrayDecode', modality='Other'),
    # dict(type='Resize', scale=(img_W, img_H), keep_ratio=False),
    dict(type='FormatShape', input_format='TCHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
train_dataloader = dict(
    batch_size=batch_size,  # videos_per_gpu
    num_workers=2,         # workers_per_gpu
    drop_last=False,
    prefetch_factor=batch_size,
    pin_memory=True,
    dataset=dict(
        type=dataset_type,
        clip_len=train_clip_len,
        frame_interval=train_frame_interval,  
        mode='train',
        
        data_root=data_root,
        pipeline=train_pipeline,
        test_mode=False,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True)
)
val_dataloader=dict(
    batch_size=batch_size,
    num_workers=2,
    drop_last=False,
    prefetch_factor=batch_size,
    pin_memory=True,
    dataset=dict(
        type=dataset_type,
        clip_len=val_clip_len,
        frame_interval=val_frame_interval, 
        mode='val', 
        
        data_root=data_root,
        pipeline=val_pipeline,
        test_mode=True,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)
)
test_dataloader=dict(
    batch_size=batch_size,
    num_workers=2,
    drop_last=False,
    prefetch_factor=batch_size,
    pin_memory=True,
    dataset=dict(
        type=dataset_type,
        clip_len=test_clip_len,
        frame_interval=test_frame_interval,  
        mode='test',
        
        data_root=data_root,
        pipeline=test_pipeline,
        test_mode=True,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)
)

val_evaluator=[
    dict(type='ErrorMetrics', metric_list=['mae', 'mse', 'rmse'], spatial_norm=True),
    dict(type='WeatherMetrics', metric_list=['nino3.4']),
]
test_evaluator=[
    dict(type='ErrorMetrics', metric_list=['mae', 'mse', 'rmse', 'mape', 'wmape'], spatial_norm=True),
    dict(type='WeatherMetrics', metric_list=['nino3.4']),
    dict(type='SimilarityMetrics', is_img=True, metric_list=['ssim', 'psnr', 'snr']),
]

data_processor=dict(
    type='VideoProcessor',
    input_len=input_len,
    output_len=output_len,
    pred_len=pred_len,
    mean=img_norm_cfg['mean'],
    std=img_norm_cfg['std'],
)