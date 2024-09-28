# dataset settings
dataset_type = 'Traffic4CastDataset'
data_root = 'data/Traffic4Cast2021'
ann_file_train = 'data/annotations/traffic4cast2021/traffic4cast2021_train.json'
ann_file_val = 'data/annotations/traffic4cast2021/traffic4cast2021_val.json'
ann_file_test = 'data/annotations/traffic4cast2021/traiifc4cast2021_test.json'
stride_train = 10
stride_val = 10
stride_test = 10


img_H = 128
img_W = 112
img_C = 8
img_norm_cfg = dict(
    mean=[0. for _ in range(img_C)], std=[255. for _ in range(img_C)], to_bgr=False
)

input_len = 9
output_len = 3
pred_len = 3

train_frame_interval=1
train_input_len=input_len
train_output_len=output_len
train_clip_len=train_input_len + train_output_len

val_frame_interval=train_frame_interval
val_input_len=input_len
val_output_len=pred_len
val_clip_len=val_input_len + val_output_len

test_frame_interval=train_frame_interval
test_input_len=input_len
test_output_len=pred_len
test_clip_len=test_input_len + test_output_len


input_shape=(input_len,img_C,img_H,img_W)
batch_size=2
train_pipeline = [
    dict(
        type='SampleFrames', 
        clip_len=train_clip_len, 
        frame_interval=train_frame_interval, 
        num_clips=1),
    dict(type='Traffic4CastDecode'),
    # dict(type='Resize', scale=(img_W, img_H), keep_ratio=False),
    dict(type='CenterCrop', crop_size=(img_W, img_H)),
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
    dict(type='Traffic4CastDecode'),
    # dict(type='Resize', scale=(img_W, img_H), keep_ratio=False),
    dict(type='CenterCrop', crop_size=(img_W, img_H)),
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
    dict(type='Traffic4CastDecode'),
    # dict(type='Resize', scale=(img_W, img_H), keep_ratio=False),
    dict(type='CenterCrop', crop_size=(img_W, img_H)),
    dict(type='FormatShape', input_format='TCHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
train_dataloader = dict(
    batch_size=batch_size, 
    num_workers=4,         
    drop_last=False,
    prefetch_factor=batch_size,
    pin_memory=True,
    dataset=dict(
        type=dataset_type,
        clip_len=train_clip_len,
        frame_interval=train_frame_interval,  
        stride=stride_train,
        
        ann_file=ann_file_train,
        data_root=data_root, 
        pipeline=train_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True)
)
val_dataloader=dict(
    batch_size=batch_size,
    num_workers=4,
    drop_last=False,
    prefetch_factor=batch_size,
    pin_memory=True,
    dataset=dict(
        type=dataset_type,
        clip_len=val_clip_len,
        frame_interval=val_frame_interval,  
        stride=stride_val,
        
        ann_file=ann_file_val,
        data_root=data_root, 
        pipeline=val_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)
)
test_dataloader=dict(
    batch_size=batch_size,
    num_workers=4,
    drop_last=False,
    prefetch_factor=batch_size,
    pin_memory=True,
    dataset=dict(
        type=dataset_type,
        clip_len=test_clip_len,
        frame_interval=test_frame_interval,  
        stride=stride_test,
        
        ann_file=ann_file_test,
        data_root=data_root, 
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)
)

val_evaluator=[
    dict(type='ErrorMetrics', metric_list=['mae', 'mse', 'rmse'], spatial_norm=True),
    dict(type='SimilarityMetrics', is_img=True, metric_list=['ssim', 'psnr', 'snr']),
]
test_evaluator=[
    dict(type='ErrorMetrics', metric_list=['mae', 'mse', 'rmse', 'mape', 'wmape'], spatial_norm=True),
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
