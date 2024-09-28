# dataset settings
dataset_type = 'RoboNetDataset'
data_root = 'data/robonet/hdf5'
ann_file_train = 'data/annotations/robonet/robonet_train.json'
ann_file_val = 'data/annotations/robonet/robonet_val.json'
ann_file_test = 'data/annotations/robonet/robonet_test.json'


img_H = 120
img_W = 160
img_C = 3
img_norm_cfg = dict(
    mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], to_bgr=False
)

input_len = 2
output_len = 10
pred_len = 10

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
batch_size=16
train_pipeline = [
    dict(
        type='SampleFrames', 
        clip_len=train_clip_len, 
        frame_interval=train_frame_interval, 
        num_clips=1),
    dict(type='RobonetDecode'),
    dict(type='Resize', scale=(img_W, img_H), keep_ratio=False),
    # dict(type='Normalize', **img_norm_cfg),
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
    dict(type='RobonetDecode'),
    dict(type='Resize', scale=(img_W, img_H), keep_ratio=False),
    # dict(type='Normalize', **img_norm_cfg),
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
    dict(type='RobonetDecode'),
    dict(type='Resize', scale=(img_W, img_H), keep_ratio=False),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='TCHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
train_dataloader = dict(
    batch_size=batch_size,  # videos_per_gpu
    num_workers=4,         # workers_per_gpu
    drop_last=False,
    prefetch_factor=batch_size,
    pin_memory=True,
    dataset=dict(
        type=dataset_type,
        clip_len=train_clip_len,
        frame_interval=train_frame_interval,  
        
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
        
        ann_file=ann_file_val, 
        data_root=data_root, 
        test_mode=True,
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
        
        ann_file=ann_file_test, 
        data_root=data_root, 
        test_mode=True,
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)
)

val_evaluator=[
    dict(type='ErrorMetrics', metric_list=['mae', 'mse', 'rmse']),
]
test_evaluator=[
    dict(type='ErrorMetrics', metric_list=['mae', 'mse', 'rmse']),
    dict(type='SimilarityMetrics', is_img=True, metric_list=['ssim', 'psnr', 'snr']),
    dict(type='PerceptionMetrics', metric_list=['lpips', 'fvd'], collect_device='gpu'),
]

data_processor=dict(
    type='VideoProcessor',
    input_len=input_len,
    output_len=output_len,
    pred_len=pred_len,
    mean=img_norm_cfg['mean'],
    std=img_norm_cfg['std'],
)