# dataset settings
dataset_type = 'SEVIRDataset'
data_root_train = 'data/sevir/train'
data_root_val = 'data/sevir/val'
data_root_test = 'data/sevir/test'

# SEVIR Dataset constants
SEVIR_DATA_TYPES = ['vis', 'ir069', 'ir107', 'vil', 'lght']
SEVIR_DATA_SHAPE = {
    'vis': (768, 768), 
    'ir069': (192, 192),
    'ir107': (192, 192),
    'vil': (384, 384),
    'lght': (48, 48), 
}
SEVIR_STD = {
    'vis': 1,  # Not utilized in original paper
    'ir069': 1174.68,
    'ir107': 2562.43,
    'vil': 47.54,
    'lght': 0.60517
}
SEVIR_MEAN = {
    'vis': 0,  # Not utilized in original paper
    'ir069': -3683.58,
    'ir107': -1552.80,
    'vil': -33.44,
    'lght': -0.02990
}
SEVIR_01_STD = {
    'vis': 1,
    'ir069': 1,
    'ir107': 1,
    'vil': 255,  # currently the only one implemented
    'lght': 1
}
SEVIR_01_MEAN = {
    'vis': 0,
    'ir069': 0,
    'ir107': 0,
    'vil': 0,  # currently the only one implemented
    'lght': 0
}

img_H, img_W = SEVIR_DATA_SHAPE['vil']  # currently the only one implemented
img_C = 1
img_norm_cfg = dict(
    mean=[SEVIR_01_MEAN['vil']], std=[SEVIR_01_STD['vil']], to_bgr=False
)

frame_interval=1
input_len=13
output_len=12
pred_len = 12


train_frame_interval=frame_interval
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
    dict(type='NPYDecode'),
    dict(type='Resize', scale=(img_W, img_H), keep_ratio=False),
    dict(type='FormatShape', input_format='TCHW'),
    dict(type='Collect', keys=['imgs',], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs',])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=val_clip_len,
        frame_interval=val_frame_interval,
        num_clips=1,
        test_mode=True),
    dict(type='NPYDecode'),
    dict(type='Resize', scale=(img_W, img_H), keep_ratio=False),
    dict(type='FormatShape', input_format='TCHW'),
    dict(type='Collect', keys=['imgs',], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs',])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=test_clip_len,
        frame_interval=test_frame_interval,
        num_clips=1,
        test_mode=True),
    dict(type='NPYDecode'),
    dict(type='Resize', scale=(img_W, img_H), keep_ratio=False),
    dict(type='FormatShape', input_format='TCHW'),
    dict(type='Collect', keys=['imgs',], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs',])
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
        
        data_root=data_root_train,
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
        
        data_root=data_root_val,
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
        
        data_root=data_root_test,
        pipeline=test_pipeline,
        test_mode=True,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)
)

val_evaluator=[
    dict(type='ErrorMetrics', metric_list=['mae', 'mse', 'rmse'], spatial_norm=True),
    dict(type='WeatherMetrics', metric_list=['bias', 'csi', 'pod', 'sucr'], threshold_list=[16, 74, 133, 160, 181, 219],),
]
test_evaluator=[
    dict(type='ErrorMetrics', metric_list=['mae', 'mse', 'rmse', 'mape', 'wmape'], spatial_norm=True),
    dict(type='SimilarityMetrics', is_img=True, metric_list=['ssim', 'psnr', 'snr']),
    dict(type='WeatherMetrics', metric_list=['bias', 'csi', 'pod', 'sucr'], threshold_list=[16, 74, 133, 160, 181, 219],),
]

data_processor=dict(
    type='VideoProcessor',
    input_len=input_len,
    output_len=output_len,
    pred_len=pred_len,
    mean=img_norm_cfg['mean'],
    std=img_norm_cfg['std'],
)