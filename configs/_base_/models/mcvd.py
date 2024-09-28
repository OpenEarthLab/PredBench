
gamma = False

final_only = True
subsample_steps = 100   # subsample in original config
clip_before = True
denoise = True

version = 'DDPM'
arch = 'unetmore'
time_conditional = True
dropout = 0.1
sigma_dist = 'linear'
sigma_begin = 0.02
sigma_end = 0.0001
num_classes = 1000
nonlinearity = 'swish'
ngf = 96
ch_mult = [1, 2, 3, 4]
num_res_blocks = 2
attn_resolutions = [8, 16, 32]
n_head_channels = 96
conditional = True
noise_in_cond = False
output_all_frames = False
cond_emb = False
spade = True
spade_dim = 128




sampler = dict(
    type='DDIMSampler',
    final_only=final_only, 
    subsample_steps=subsample_steps, 
    clip_before=clip_before, 
    denoise=denoise, 
    t_min=-1, 
    gamma=gamma
)
score_net = dict(
    type='UNetMore_DDPM',
    input_output_shape=(10, 10, 3, 64, 64), #(T_in, T_out, C, H, W)
    version=version, 
    spade=spade, 
    gamma=gamma, 
    noise_in_cond=noise_in_cond, 
    nonlinearity=nonlinearity, 
    num_classes=num_classes, 
    sigma_dist=sigma_dist, 
    sigma_begin=sigma_begin, 
    sigma_end=sigma_end, 
    device='cuda', arch=arch, 
    ngf=ngf, 
    ch_mult=ch_mult, 
    num_res_blocks=num_res_blocks, 
    attn_resolutions=attn_resolutions, 
    dropout=dropout, 
    time_conditional=time_conditional, 
    cond_emb=cond_emb, 
    spade_dim=spade_dim, 
    n_head_channels=n_head_channels, 
    output_all_frames=output_all_frames,
)