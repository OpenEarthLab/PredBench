
base_units = 64
block_units = None
scale_alpha = 1.0

enc_depth = [1, 1]
dec_depth = [1, 1]
enc_use_inter_ffn = True
dec_use_inter_ffn = True
dec_hierarchical_pos_embed = False

downsample = 2
downsample_type = 'patch_merge'
upsample_type = 'upsample'

# global vectors
num_global_vectors = 0
use_dec_self_global = False
dec_self_update_global = True
use_dec_cross_global = False
use_global_vector_ffn = False
use_global_self_attn = False
separate_global_qkv = False
global_dim_ratio = 1

self_pattern = 'axial'
cross_self_pattern = 'axial'
cross_pattern = 'cross_1x1'
dec_cross_last_n_frames = None

attn_drop = 0.1
proj_drop = 0.1
ffn_drop = 0.1
num_heads = 4

ffn_activation = 'gelu'
gated_ffn = False
norm_layer = 'layer_norm'
padding_type = 'zeros'
pos_embed_type = 't+h+w'
use_relative_pos = True
self_attn_use_final_proj = True
dec_use_first_self_attn = False

z_init_method = 'zeros'
initial_downsample_type = 'conv'
initial_downsample_activation = 'leaky'
initial_downsample_scale = [1, 1, 2]
initial_downsample_conv_layers = 2
final_upsample_conv_layers = 1
checkpoint_level = 0

attn_linear_init_mode = '0'
ffn_linear_init_mode = '0'
conv_init_mode = '0'
down_up_linear_init_mode = '0'
norm_init_mode = '0'


transformer=dict(
    type='CuboidTransformerModel',
    input_shape=(10, 64, 64, 3),    # (T, H, W, C)
    target_shape=(10, 64, 64, 3),   # (T, H, W, C)
    base_units=base_units,
    block_units=block_units,
    scale_alpha=scale_alpha,
    enc_depth=enc_depth,
    dec_depth=dec_depth,
    enc_use_inter_ffn=enc_use_inter_ffn,
    dec_use_inter_ffn=dec_use_inter_ffn,
    dec_hierarchical_pos_embed=dec_hierarchical_pos_embed,
    downsample=downsample,
    downsample_type=downsample_type,
    enc_attn_patterns = [self_pattern] * len(enc_depth),
    dec_self_attn_patterns = [cross_self_pattern] * len(dec_depth),
    dec_cross_attn_patterns = [cross_pattern] * len(dec_depth),
    dec_cross_last_n_frames=dec_cross_last_n_frames,
    dec_use_first_self_attn=dec_use_first_self_attn,
    num_heads=num_heads,
    attn_drop=attn_drop,
    proj_drop=proj_drop,
    ffn_drop=ffn_drop,
    upsample_type=upsample_type,
    ffn_activation=ffn_activation,
    gated_ffn=gated_ffn,
    norm_layer=norm_layer,
    # global vectors
    num_global_vectors=num_global_vectors,
    use_dec_self_global=use_dec_self_global,
    dec_self_update_global=dec_self_update_global,
    use_dec_cross_global=use_dec_cross_global,
    use_global_vector_ffn=use_global_vector_ffn,
    use_global_self_attn=use_global_self_attn,
    separate_global_qkv=separate_global_qkv,
    global_dim_ratio=global_dim_ratio,
    # initial_downsample
    initial_downsample_type=initial_downsample_type,
    initial_downsample_activation=initial_downsample_activation,
    # initial_downsample_type=="conv"
    initial_downsample_scale=initial_downsample_scale,
    initial_downsample_conv_layers=initial_downsample_conv_layers,
    final_upsample_conv_layers=final_upsample_conv_layers,
    # misc
    padding_type=padding_type,
    z_init_method=z_init_method,
    checkpoint_level=checkpoint_level,
    pos_embed_type=pos_embed_type,
    use_relative_pos=use_relative_pos,
    self_attn_use_final_proj=self_attn_use_final_proj,
    # initialization
    attn_linear_init_mode=attn_linear_init_mode,
    ffn_linear_init_mode=ffn_linear_init_mode,
    conv_init_mode=conv_init_mode,
    down_up_linear_init_mode=down_up_linear_init_mode,
    norm_init_mode=norm_init_mode,
)