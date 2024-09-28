SINGLE_VARIABLES = {
    't2m': '2m_temperature',                # Temperature at 2 m height above surface           [K]
    'u10': '10m_u_component_of_wind',       # Wind in x/longitude-direction at 10 m height      [m s−1]
    'v10': '10m_v_component_of_wind',       # Wind in y/latitude-direction at 10 m height       [m s−1]
    'tcc': 'total_cloud_cover',             # Fractional cloud cover                            (0–1)
    'tp': 'total_precipitation',            # Hourly precipitation                              [m]
    'tisr': 'toa_incident_solar_radiation', # Accumulated hourly incident solar radiation       [J m−2]
}
MULTI_VARIABLES = {
    'z': 'geopotential',                    # Proportional to the height of a pressure level    [m2s−2]
    't': 'temperature',                     # Temperature                                       [K]
    'q': 'specific_humidity',               # Mixing ratio of water vapor                       [kg kg−1]
    'r': 'relative_humidity',               # Humidity relative to saturation                   [%]
    'u': 'u_component_of_wind',             # Wind in x/longitude-direction                     [m s−1]
    'v': 'v_component_of_wind',             # Wind in y/latitude direction                      [m s−1]
    'vo': 'vorticity',                      # Relative horizontal vorticity                     [1 s−1]
    'pv': 'potential_vorticity',            # Potential vorticity                               [K m2 kg−1 s−1]
}
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
variable_description = {    # variable for 69 variables: t2m, u10, v10, tp, z, t, r, u, v
    "t2m": {
        "channel": 0,
        "mean": 278.4564870856443,
        "std": 21.265027650260755
    },
    "u10": {
        "channel": 1,
        "mean": -0.0530283683368791,
        "std": 5.540960562031698
    },
    "v10": {
        "channel": 2,
        "mean": 0.18636863377777407,
        "std": 4.761945612664701
    },
    "tp": {
        "channel": 3,
        "mean": 9.984565030612924e-05,
        "std": 0.0003746203863219765
    },
    "z_50": {
        "channel": 4,
        "mean": 199352.57820299748,
        "std": 5879.004870527404
    },
    "z_100": {
        "channel": 5,
        "mean": 157621.15846563203,
        "std": 5505.050026218838
    },
    "z_150": {
        "channel": 6,
        "mean": 133121.747558806,
        "std": 5836.781124207325
    },
    "z_200": {
        "channel": 7,
        "mean": 115312.9521689395,
        "std": 5824.0431667247085
    },
    "z_250": {
        "channel": 8,
        "mean": 101208.96428009825,
        "std": 5531.3351715192575
    },
    "z_300": {
        "channel": 9,
        "mean": 89402.02720077451,
        "std": 5086.688230980537
    },
    "z_400": {
        "channel": 10,
        "mean": 69971.11520708908,
        "std": 4150.618931776315
    },
    "z_500": {
        "channel": 11,
        "mean": 54108.17928708192,
        "std": 3354.7553269562677
    },
    "z_600": {
        "channel": 12,
        "mean": 40643.06247051602,
        "std": 2695.5869661518964
    },
    "z_700": {
        "channel": 13,
        "mean": 28925.294253674496,
        "std": 2135.319483622276
    },
    "z_850": {
        "channel": 14,
        "mean": 13748.344600698667,
        "std": 1468.4257655700098
    },
    "z_925": {
        "channel": 15,
        "mean": 7014.361392289925,
        "std": 1227.114274255912
    },
    "z_1000": {
        "channel": 16,
        "mean": 738.2807581190701,
        "std": 1070.506228373003
    },
    "t_50": {
        "channel": 17,
        "mean": 212.49501509269894,
        "std": 10.275156572873254
    },
    "t_100": {
        "channel": 18,
        "mean": 208.3946397793598,
        "std": 12.518752873104798
    },
    "t_150": {
        "channel": 19,
        "mean": 213.2947049909197,
        "std": 8.939279791503143
    },
    "t_200": {
        "channel": 20,
        "mean": 218.0289570929518,
        "std": 7.190521251818904
    },
    "t_250": {
        "channel": 21,
        "mean": 222.73254935574673,
        "std": 8.528138592904716
    },
    "t_300": {
        "channel": 22,
        "mean": 228.83115062174505,
        "std": 10.71310670353003
    },
    "t_400": {
        "channel": 23,
        "mean": 242.09947692004096,
        "std": 12.701194067815829
    },
    "t_500": {
        "channel": 24,
        "mean": 252.91528895041756,
        "std": 13.071147057611798
    },
    "t_600": {
        "channel": 25,
        "mean": 261.0999571422843,
        "std": 13.416323808762652
    },
    "t_700": {
        "channel": 26,
        "mean": 267.35783957031265,
        "std": 14.779978233491846
    },
    "t_850": {
        "channel": 27,
        "mean": 274.5183571311846,
        "std": 15.60257554579701
    },
    "t_925": {
        "channel": 28,
        "mean": 277.31681641173003,
        "std": 16.11263285629241
    },
    "t_1000": {
        "channel": 29,
        "mean": 280.97190507170177,
        "std": 17.155902960638247
    },
    "r_50": {
        "channel": 30,
        "mean": 6.511358968728509,
        "std": 15.529836191063442
    },
    "r_100": {
        "channel": 31,
        "mean": 26.264878314150117,
        "std": 33.448527944235714
    },
    "r_150": {
        "channel": 32,
        "mean": 26.7949989632441,
        "std": 32.22046881917166
    },
    "r_200": {
        "channel": 33,
        "mean": 35.665004420665426,
        "std": 34.1015038567506
    },
    "r_250": {
        "channel": 34,
        "mean": 47.299001650330354,
        "std": 34.41553746432257
    },
    "r_300": {
        "channel": 35,
        "mean": 53.845128422836055,
        "std": 33.870575925938965
    },
    "r_400": {
        "channel": 36,
        "mean": 52.585173542914724,
        "std": 34.092882653729944
    },
    "r_500": {
        "channel": 37,
        "mean": 50.38509777109044,
        "std": 33.47841131762907
    },
    "r_600": {
        "channel": 38,
        "mean": 51.56973879147407,
        "std": 32.64479278270746
    },
    "r_700": {
        "channel": 39,
        "mean": 54.95396485898575,
        "std": 31.381964516288175
    },
    "r_850": {
        "channel": 40,
        "mean": 69.12254820722166,
        "std": 26.295896374623183
    },
    "r_925": {
        "channel": 41,
        "mean": 79.09543472831598,
        "std": 21.468516598458567
    },
    "r_1000": {
        "channel": 42,
        "mean": 78.60669913883737,
        "std": 18.21901415271577
    },
    "u_50": {
        "channel": 43,
        "mean": 5.634260007777595,
        "std": 15.318779011446775
    },
    "u_100": {
        "channel": 44,
        "mean": 10.268632338805652,
        "std": 13.517099389326702
    },
    "u_150": {
        "channel": 45,
        "mean": 13.534361833392937,
        "std": 16.0300443606774
    },
    "u_200": {
        "channel": 46,
        "mean": 14.209449837202342,
        "std": 17.66140414898396
    },
    "u_250": {
        "channel": 47,
        "mean": 13.348236932058189,
        "std": 17.955196960082958
    },
    "u_300": {
        "channel": 48,
        "mean": 11.80463621623413,
        "std": 17.109722606861304
    },
    "u_400": {
        "channel": 49,
        "mean": 8.817827706395354,
        "std": 14.334814041621819
    },
    "u_500": {
        "channel": 50,
        "mean": 6.561363374459267,
        "std": 11.976260520664134
    },
    "u_600": {
        "channel": 51,
        "mean": 4.812414306248437,
        "std": 10.32541839127276
    },
    "u_700": {
        "channel": 52,
        "mean": 3.3435517722262738,
        "std": 9.158075108319668
    },
    "u_850": {
        "channel": 53,
        "mean": 1.413628674629966,
        "std": 8.179482993466186
    },
    "u_925": {
        "channel": 54,
        "mean": 0.6133454894506712,
        "std": 7.930190481030738
    },
    "u_1000": {
        "channel": 55,
        "mean": -0.03533602561696744,
        "std": 6.134853230873978
    },
    "v_50": {
        "channel": 56,
        "mean": 0.004142794736935441,
        "std": 7.040770427987679
    },
    "v_100": {
        "channel": 57,
        "mean": 0.014410842271195152,
        "std": 7.472195170243973
    },
    "v_150": {
        "channel": 58,
        "mean": -0.03666761547508155,
        "std": 9.56579864486826
    },
    "v_200": {
        "channel": 59,
        "mean": -0.04472469947340962,
        "std": 11.872408859449058
    },
    "v_250": {
        "channel": 60,
        "mean": -0.029776500860700526,
        "std": 13.374472277067502
    },
    "v_300": {
        "channel": 61,
        "mean": -0.02294325706302894,
        "std": 13.33729358279563
    },
    "v_400": {
        "channel": 62,
        "mean": -0.01773763967263352,
        "std": 11.227189986236919
    },
    "v_500": {
        "channel": 63,
        "mean": -0.024035272796776603,
        "std": 9.178088213021436
    },
    "v_600": {
        "channel": 64,
        "mean": -0.027187391977621357,
        "std": 7.799095387337467
    },
    "v_700": {
        "channel": 65,
        "mean": 0.020306812725551307,
        "std": 6.866160594882073
    },
    "v_850": {
        "channel": 66,
        "mean": 0.142814150161757,
        "std": 6.2587639682841685
    },
    "v_925": {
        "channel": 67,
        "mean": 0.20342171875493803,
        "std": 6.46473584751148
    },
    "v_1000": {
        "channel": 68,
        "mean": 0.18407060384821664,
        "std": 5.304272758031071
    }
}
latitude = [
    -89.296875,
    -87.890625,
    -86.484375,
    -85.078125,
    -83.671875,
    -82.265625,
    -80.859375,
    -79.453125,
    -78.046875,
    -76.640625,
    -75.234375,
    -73.828125,
    -72.421875,
    -71.015625,
    -69.609375,
    -68.203125,
    -66.796875,
    -65.390625,
    -63.984375,
    -62.578125,
    -61.171875,
    -59.765625,
    -58.359375,
    -56.953125,
    -55.546875,
    -54.140625,
    -52.734375,
    -51.328125,
    -49.921875,
    -48.515625,
    -47.109375,
    -45.703125,
    -44.296875,
    -42.890625,
    -41.484375,
    -40.078125,
    -38.671875,
    -37.265625,
    -35.859375,
    -34.453125,
    -33.046875,
    -31.640625,
    -30.234375,
    -28.828125,
    -27.421875,
    -26.015625,
    -24.609375,
    -23.203125,
    -21.796875,
    -20.390625,
    -18.984375,
    -17.578125,
    -16.171875,
    -14.765625,
    -13.359375,
    -11.953125,
    -10.546875,
    -9.140625,
    -7.734375,
    -6.328125,
    -4.921875,
    -3.515625,
    -2.109375,
    -0.703125,
    0.703125,
    2.109375,
    3.515625,
    4.921875,
    6.328125,
    7.734375,
    9.140625,
    10.546875,
    11.953125,
    13.359375,
    14.765625,
    16.171875,
    17.578125,
    18.984375,
    20.390625,
    21.796875,
    23.203125,
    24.609375,
    26.015625,
    27.421875,
    28.828125,
    30.234375,
    31.640625,
    33.046875,
    34.453125,
    35.859375,
    37.265625,
    38.671875,
    40.078125,
    41.484375,
    42.890625,
    44.296875,
    45.703125,
    47.109375,
    48.515625,
    49.921875,
    51.328125,
    52.734375,
    54.140625,
    55.546875,
    56.953125,
    58.359375,
    59.765625,
    61.171875,
    62.578125,
    63.984375,
    65.390625,
    66.796875,
    68.203125,
    69.609375,
    71.015625,
    72.421875,
    73.828125,
    75.234375,
    76.640625,
    78.046875,
    79.453125,
    80.859375,
    82.265625,
    83.671875,
    85.078125,
    86.484375,
    87.890625,
    89.296875
]
# dataset settings
dataset_type = 'WeatherBenchDatasetNp'
data_root = 'data/weatherbench128x256_npy69'
clim_path = 'data/weatherbench_clim_1993_2016'
year_split_train = [1979, 2015]     # 1979 <= year <= 2015
year_split_val = [2016, 2016]       # year = 2016
year_split_test = [2017, 2018]      # 2017 <= year <= 2018
hour_stride_train = 6
hour_stride_val = 3
hour_stride_test = 3

# used_variables = ['tp']    
used_variables = ['t2m', 'u10', 'v10', 'tp']    
for v in ['z', 't', 'r', 'u', 'v']:
    for level in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]:
        used_variables.append(f'{v}_{level}')
used_channels = [variable_description[v]['channel'] for v in used_variables]
metric_variables = ['t2m', 'z_500', 't_850']
metric_channels = [variable_description[v]['channel'] for v in metric_variables]


img_H = 128
img_W = 256
img_C = len(used_variables)
img_norm_cfg = dict(
    mean=[variable_description[v]['mean'] for v in used_variables], 
    std=[variable_description[v]['std'] for v in used_variables], 
    to_bgr=False
)

input_len = 2
output_len = 1
pred_len = 1

train_frame_interval=6
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
    # dict(type='WeatherBench69Decode'),
    dict(type='RawNPYDecode'),
    # dict(type='Resize', scale=(img_W, img_H), keep_ratio=False),
    # dict(type='FormatShape', input_format='TCHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=[]),
    dict(type='AccelerateNumpyToTensor', keys=['imgs'], channels=used_channels)
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=val_clip_len,
        frame_interval=val_frame_interval,
        num_clips=1,
        test_mode=True),
    # dict(type='WeatherBench69Decode'),
    dict(type='RawNPYDecode', clim_path=clim_path, metric_variables=metric_variables),
    # dict(type='Resize', scale=(img_W, img_H), keep_ratio=False),
    # dict(type='FormatShape', input_format='TCHW'),
    dict(type='Collect', keys=['imgs', 'clims'], meta_keys=[]),
    dict(type='AccelerateNumpyToTensor', keys=['imgs'], channels=used_channels),
    dict(type='ToTensor', keys=['clims'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=test_clip_len,
        frame_interval=test_frame_interval,
        num_clips=1,
        test_mode=True),
    # dict(type='WeatherBench69Decode'),
    dict(type='RawNPYDecode', clim_path=clim_path, metric_variables=metric_variables),
    # dict(type='Resize', scale=(img_W, img_H), keep_ratio=False),
    # dict(type='FormatShape', input_format='TCHW'),
    dict(type='Collect', keys=['imgs', 'clims'], meta_keys=[]),
    dict(type='AccelerateNumpyToTensor', keys=['imgs'], channels=used_channels),
    dict(type='ToTensor', keys=['clims'])
]
train_dataloader = dict(
    batch_size=batch_size, 
    num_workers=2,         
    drop_last=False,
    prefetch_factor=batch_size,
    pin_memory=True,
    dataset=dict(
        type=dataset_type,
        clip_len=train_clip_len,
        frame_interval=train_frame_interval,  
        year_split=year_split_train,
        hour_stride=hour_stride_train,
        
        data_root=data_root, 
        pipeline=train_pipeline,
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
        year_split=year_split_val,
        hour_stride=hour_stride_val,
        
        data_root=data_root, 
        pipeline=val_pipeline,
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
        year_split=year_split_test,
        hour_stride=hour_stride_test,
        
        data_root=data_root, 
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)
)

val_evaluator=[
    dict(type='ErrorMetrics', metric_list=['mae', 'mse', 'rmse']),
    dict(type='WeatherMetrics', metric_list=['wmae', 'wmse', 'wrmse', 'acc'], latitude=latitude, metric_channels=metric_channels),
]
test_evaluator=[
    dict(type='ErrorMetrics', metric_list=['mae', 'mse', 'rmse']),
    dict(type='WeatherMetrics', metric_list=['wmae', 'wmse', 'wrmse', 'acc'], latitude=latitude, metric_channels=metric_channels),
]


data_processor=dict(
    type='VideoProcessor',
    input_len=input_len,
    output_len=output_len,
    pred_len=pred_len,
    mean=img_norm_cfg['mean'],
    std=img_norm_cfg['std'],
)