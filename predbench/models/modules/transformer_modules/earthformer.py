
"""A space-time Transformer with Cuboid Attention"""
from typing import Sequence, Union
import torch
from torch import nn
import torch.nn.functional as F
from .cuboid_transformer_patterns import CuboidSelfAttentionPatterns, CuboidCrossAttentionPatterns
from .cuboid_transformer_modules import (
    PosEmbed, PatchMerging3D, Upsample3DLayer, 
    StackCuboidSelfAttentionBlock, StackCuboidCrossAttentionBlock
)
from .utils import (get_activation, apply_initialization, round_to)

from predbench.registry import MODELS

class CuboidTransformerEncoder(nn.Module):
    """Encoder of the CuboidTransformer

    x --> attn_block --> patch_merge --> attn_block --> patch_merge --> ... --> out

    """
    def __init__(self,
                 input_shape,
                 base_units=128,
                 block_units=None,
                 scale_alpha=1.0,
                 depth=[4, 4, 4],
                 downsample=2,
                 downsample_type='patch_merge',
                 block_attn_patterns=None,
                 block_cuboid_size=[(4, 4, 4),
                                    (4, 4, 4)],
                 block_strategy=[('l', 'l', 'l'),
                                 ('d', 'd', 'd')],
                 block_shift_size=[(0, 0, 0),
                                   (0, 0, 0)],
                 num_heads=4,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 ffn_drop=0.0,
                 activation="leaky",
                 ffn_activation='leaky',
                 gated_ffn=False,
                 norm_layer='layer_norm',
                 use_inter_ffn=True,
                 padding_type='ignore',
                 checkpoint_level=True,
                 use_relative_pos=True,
                 self_attn_use_final_proj=True,
                 # global vectors
                 use_global_vector=False,
                 use_global_vector_ffn=True,
                 use_global_self_attn=False,
                 separate_global_qkv=False,
                 global_dim_ratio=1,
                 # initialization
                 attn_linear_init_mode="0",
                 ffn_linear_init_mode="0",
                 conv_init_mode="0",
                 down_linear_init_mode="0",
                 norm_init_mode="0",
                 ):
        """

        Parameters
        ----------
        input_shape
            The shape of the input. Contains T, H, W, C
        initial_data_thw
            The shape of the first layer
        base_units
            The number of units
        scale_alpha
            We scale up the channels based on the formula:
            - round_to(base_units * max(downsample_scale) ** units_alpha, 4)
        depth
            The number of layers for each block
        downsample
            The downsample ratio
        downsample_type
            Type of the downsampling layer
        block_attn_patterns
            Attention pattern for the cuboid attention for each block.
        block_cuboid_size
            A list of cuboid size parameters
        block_strategy
            A list of cuboid strategies
        block_shift_size
            A list of shift sizes
        num_global
            The number of global vectors
        num_heads
            The number of heads.
        attn_drop
        proj_drop
        ffn_drop
        gated_ffn
            Whether to enable gated ffn or not
        norm_layer
            The normalization layer
        use_inter_ffn
            Whether to use intermediate FFN
        padding_type
        """
        super(CuboidTransformerEncoder, self).__init__()
        # initialization mode
        # self.attn_linear_init_mode = attn_linear_init_mode
        # self.ffn_linear_init_mode = ffn_linear_init_mode
        # self.conv_init_mode = conv_init_mode
        self.down_linear_init_mode = down_linear_init_mode
        # self.norm_init_mode = norm_init_mode

        self.input_shape = input_shape
        self.depth = depth
        self.num_blocks = len(depth)
        self.base_units = base_units
        self.scale_alpha = scale_alpha
        if not isinstance(downsample, (tuple, list)):
            downsample = (1, downsample, downsample)
        self.downsample = downsample
        self.downsample_type = downsample_type
        self.num_heads = num_heads
        self.use_global_vector = use_global_vector
        self.checkpoint_level = checkpoint_level
        if block_units is None:
            block_units = [round_to(base_units * int((max(downsample) ** scale_alpha) ** i), 4)
                           for i in range(self.num_blocks)]
        else:
            assert len(block_units) == self.num_blocks and block_units[0] == base_units
        self.block_units = block_units


        if self.num_blocks > 1:
            if downsample_type == 'patch_merge':
                self.down_layers = nn.ModuleList(
                    [PatchMerging3D(dim=self.block_units[i],
                                    downsample=downsample,
                                    # downsample=(1, 1, 1),
                                    padding_type=padding_type,
                                    out_dim=self.block_units[i + 1],
                                    linear_init_mode=down_linear_init_mode,
                                    norm_init_mode=norm_init_mode)
                     for i in range(self.num_blocks - 1)])
            else:
                raise NotImplementedError
            if self.use_global_vector:
                self.down_layer_global_proj = nn.ModuleList(
                    [nn.Linear(in_features=global_dim_ratio*self.block_units[i],
                               out_features=global_dim_ratio*self.block_units[i + 1])
                     for i in range(self.num_blocks - 1)])

        if block_attn_patterns is not None:
            mem_shapes = self.get_mem_shapes()
            if isinstance(block_attn_patterns, (tuple, list)):
                assert len(block_attn_patterns) == self.num_blocks
            else:
                block_attn_patterns = [block_attn_patterns for _ in range(self.num_blocks)]
            block_cuboid_size = []
            block_strategy = []
            block_shift_size = []
            for idx, key in enumerate(block_attn_patterns):
                # func = CuboidSelfAttentionPatterns.get(key)
                # cuboid_size, strategy, shift_size = func(mem_shapes[idx])
                cuboid_size, strategy, shift_size = CuboidSelfAttentionPatterns.build(
                    dict(type=key, input_shape=mem_shapes[idx])
                )
                block_cuboid_size.append(cuboid_size)
                block_strategy.append(strategy)
                block_shift_size.append(shift_size)
        else:
            if not isinstance(block_cuboid_size[0][0], (list, tuple)):
                block_cuboid_size = [block_cuboid_size for _ in range(self.num_blocks)]
            else:
                assert len(block_cuboid_size) == self.num_blocks,\
                    f'Incorrect input format! Received block_cuboid_size={block_cuboid_size}'

            if not isinstance(block_strategy[0][0], (list, tuple)):
                block_strategy = [block_strategy for _ in range(self.num_blocks)]
            else:
                assert len(block_strategy) == self.num_blocks,\
                    f'Incorrect input format! Received block_strategy={block_strategy}'

            if not isinstance(block_shift_size[0][0], (list, tuple)):
                block_shift_size = [block_shift_size for _ in range(self.num_blocks)]
            else:
                assert len(block_shift_size) == self.num_blocks,\
                    f'Incorrect input format! Received block_shift_size={block_shift_size}'
        # self.block_cuboid_size = block_cuboid_size
        # self.block_strategy = block_strategy
        # self.block_shift_size = block_shift_size

        self.blocks = nn.ModuleList([nn.Sequential(
            *[StackCuboidSelfAttentionBlock(
                dim=self.block_units[i],
                num_heads=num_heads,
                block_cuboid_size=block_cuboid_size[i],
                block_strategy=block_strategy[i],
                block_shift_size=block_shift_size[i],
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                ffn_drop=ffn_drop,
                activation=ffn_activation,
                gated_ffn=gated_ffn,
                norm_layer=norm_layer,
                use_inter_ffn=use_inter_ffn,
                padding_type=padding_type,
                use_global_vector=use_global_vector,
                use_global_vector_ffn=use_global_vector_ffn,
                use_global_self_attn=use_global_self_attn,
                separate_global_qkv=separate_global_qkv,
                global_dim_ratio=global_dim_ratio,
                checkpoint_level=checkpoint_level,
                use_relative_pos=use_relative_pos,
                use_final_proj=self_attn_use_final_proj,
                # initialization
                attn_linear_init_mode=attn_linear_init_mode,
                ffn_linear_init_mode=ffn_linear_init_mode,
                norm_init_mode=norm_init_mode,
            ) for _ in range(depth[i])])
            for i in range(self.num_blocks)])
        self.reset_parameters()

    def reset_parameters(self):
        if self.num_blocks > 1:
            for m in self.down_layers:
                m.reset_parameters()
            if self.use_global_vector:
                apply_initialization(self.down_layer_global_proj,
                                     linear_mode=self.down_linear_init_mode)
        for ms in self.blocks:
            for m in ms:
                m.reset_parameters()

    def get_mem_shapes(self):
        """Get the shape of the output memory based on the input shape. This can be used for constructing the decoder.

        Returns
        -------
        mem_shapes
            A list of shapes of the output memory
        """

        if self.num_blocks == 1:
            return [self.input_shape]
        else:
            mem_shapes = [self.input_shape]
            curr_shape = self.input_shape
            for down_layer in self.down_layers:
                curr_shape = down_layer.get_out_shape(curr_shape)
                mem_shapes.append(curr_shape)
            return mem_shapes

    def forward(self, x, global_vectors=None):
        """

        Parameters
        ----------
        x
            Shape (B, T, H, W, C)

        Returns
        -------
        out
            A list of tensors from the bottom layer to the top layer of the encoder. For example, it can have shape
            - (B, T, H, W, C1)
            - (B, T, H // 2, W // 2, 2 * C1)
            - (B, T, H // 4, W // 4, 4 * C1)
            ...
        global_mem_out
            Optional
        """
        B, T, H, W, C_in = x.shape
        assert (T, H, W, C_in) == self.input_shape

        if self.use_global_vector:
            out = []
            global_mem_out = []
            for i in range(self.num_blocks):
                for l in self.blocks[i]:
                    x, global_vectors = l(x, global_vectors)
                out.append(x)
                global_mem_out.append(global_vectors)
                if self.num_blocks > 1 and i < self.num_blocks - 1:
                    x = self.down_layers[i](x)
                    global_vectors = self.down_layer_global_proj[i](global_vectors)
            return out, global_mem_out
        else:
            out = []
            for i in range(self.num_blocks):
                x = self.blocks[i](x)
                out.append(x)
                if self.num_blocks > 1 and i < self.num_blocks - 1:
                    x = self.down_layers[i](x)
            return out


class CuboidTransformerDecoder(nn.Module):
    """Decoder of the CuboidTransformer.

    For each block, we first apply the StackCuboidSelfAttention and then apply the StackCuboidCrossAttention

    Repeat the following structure K times

        x --> StackCuboidSelfAttention --> |
                                           |----> StackCuboidCrossAttention (If used) --> out
                                   mem --> |

    """
    def __init__(self,
                 target_temporal_length,
                 mem_shapes,
                 cross_start=0,
                 depth=[2, 2],
                 upsample_type="upsample",
                 upsample_kernel_size=3,
                 block_self_attn_patterns=None,
                 block_self_cuboid_size=[(4, 4, 4), (4, 4, 4)],
                 block_self_cuboid_strategy=[('l', 'l', 'l'), ('d', 'd', 'd')],
                 block_self_shift_size=[(1, 1, 1), (0, 0, 0)],
                 block_cross_attn_patterns=None,
                 block_cross_cuboid_hw=[(4, 4), (4, 4)],
                 block_cross_cuboid_strategy=[('l', 'l', 'l'), ('d', 'l', 'l')],
                 block_cross_shift_hw=[(0, 0), (0, 0)],
                 block_cross_n_temporal=[1, 2],
                 cross_last_n_frames=None,
                 num_heads=4,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 ffn_drop=0.0,
                 ffn_activation='leaky',
                 gated_ffn=False,
                 norm_layer='layer_norm',
                 use_inter_ffn=False,
                 hierarchical_pos_embed=False,
                 pos_embed_type='t+hw',
                 max_temporal_relative=50,
                 padding_type='ignore',
                 checkpoint_level=True,
                 use_relative_pos=True,
                 self_attn_use_final_proj=True,
                 use_first_self_attn=False,
                 # global vectors
                 use_self_global=False,
                 self_update_global=True,
                 use_cross_global=False,
                 use_global_vector_ffn=True,
                 use_global_self_attn=False,
                 separate_global_qkv=False,
                 global_dim_ratio=1,
                 # initialization
                 attn_linear_init_mode="0",
                 ffn_linear_init_mode="0",
                 conv_init_mode="0",
                 up_linear_init_mode="0",
                 norm_init_mode="0",
                 ):
        """

        Parameters
        ----------
        target_temporal_length
        mem_shapes
        cross_start
            The block to start cross attention
        depth
            Depth of each block
        upsample_type
            The type of the upsampling layers
        upsample_kernel_size
        block_self_attn_patterns
            Pattern of the block self attentions
        block_self_cuboid_size
        block_self_cuboid_strategy
        block_self_shift_size
        block_cross_attn_patterns
        block_cross_cuboid_hw
        block_cross_cuboid_strategy
        block_cross_shift_hw
        block_cross_n_temporal
        num_heads
        attn_drop
        proj_drop
        ffn_drop
        ffn_activation
        gated_ffn
        norm_layer
        use_inter_ffn
        hierarchical_pos_embed
            Whether to add pos embedding for each hierarchy.
        max_temporal_relative
        padding_type
        checkpoint_level
        """
        super(CuboidTransformerDecoder, self).__init__()
        # initialization mode
        # self.attn_linear_init_mode = attn_linear_init_mode
        # self.ffn_linear_init_mode = ffn_linear_init_mode
        # self.conv_init_mode = conv_init_mode
        # self.up_linear_init_mode = up_linear_init_mode
        # self.norm_init_mode = norm_init_mode

        assert len(depth) == len(mem_shapes)
        self.target_temporal_length = target_temporal_length
        self.num_blocks = len(mem_shapes)
        self.cross_start = cross_start
        self.mem_shapes = mem_shapes
        self.depth = depth
        self.upsample_type = upsample_type
        self.hierarchical_pos_embed = hierarchical_pos_embed
        self.checkpoint_level = checkpoint_level
        self.use_self_global = use_self_global
        self.self_update_global = self_update_global
        self.use_cross_global = use_cross_global
        self.use_global_vector_ffn = use_global_vector_ffn
        self.use_first_self_attn = use_first_self_attn
        if block_self_attn_patterns is not None:
            if isinstance(block_self_attn_patterns, (tuple, list)):
                assert len(block_self_attn_patterns) == self.num_blocks
            else:
                block_self_attn_patterns = [block_self_attn_patterns for _ in range(self.num_blocks)]
            block_self_cuboid_size = []
            block_self_cuboid_strategy = []
            block_self_shift_size = []
            for idx, key in enumerate(block_self_attn_patterns):
                # func = CuboidSelfAttentionPatterns.get(key)
                # cuboid_size, strategy, shift_size = func(mem_shapes[idx])
                cuboid_size, strategy, shift_size = CuboidSelfAttentionPatterns.build(
                    dict(type=key, input_shape=mem_shapes[idx])
                )
                block_self_cuboid_size.append(cuboid_size)
                block_self_cuboid_strategy.append(strategy)
                block_self_shift_size.append(shift_size)
        else:
            if not isinstance(block_self_cuboid_size[0][0], (list, tuple)):
                block_self_cuboid_size = [block_self_cuboid_size for _ in range(self.num_blocks)]
            else:
                assert len(block_self_cuboid_size) == self.num_blocks,\
                    f'Incorrect input format! Received block_self_cuboid_size={block_self_cuboid_size}'

            if not isinstance(block_self_cuboid_strategy[0][0], (list, tuple)):
                block_self_cuboid_strategy = [block_self_cuboid_strategy for _ in range(self.num_blocks)]
            else:
                assert len(block_self_cuboid_strategy) == self.num_blocks,\
                    f'Incorrect input format! Received block_self_cuboid_strategy={block_self_cuboid_strategy}'

            if not isinstance(block_self_shift_size[0][0], (list, tuple)):
                block_self_shift_size = [block_self_shift_size for _ in range(self.num_blocks)]
            else:
                assert len(block_self_shift_size) == self.num_blocks,\
                    f'Incorrect input format! Received block_self_shift_size={block_self_shift_size}'
        self_blocks = []
        for i in range(self.num_blocks):
            if not self.use_first_self_attn and i == self.num_blocks - 1:
                # For the top block, we won't use an additional self attention layer.
                ele_depth = depth[i] - 1
            else:
                ele_depth = depth[i]
            stack_cuboid_blocks =\
                [StackCuboidSelfAttentionBlock(
                    dim=self.mem_shapes[i][-1],
                    num_heads=num_heads,
                    block_cuboid_size=block_self_cuboid_size[i],
                    block_strategy=block_self_cuboid_strategy[i],
                    block_shift_size=block_self_shift_size[i],
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    ffn_drop=ffn_drop,
                    activation=ffn_activation,
                    gated_ffn=gated_ffn,
                    norm_layer=norm_layer,
                    use_inter_ffn=use_inter_ffn,
                    padding_type=padding_type,
                    use_global_vector=use_self_global,
                    use_global_vector_ffn=use_global_vector_ffn,
                    use_global_self_attn=use_global_self_attn,
                    separate_global_qkv=separate_global_qkv,
                    global_dim_ratio=global_dim_ratio,
                    checkpoint_level=checkpoint_level,
                    use_relative_pos=use_relative_pos,
                    use_final_proj=self_attn_use_final_proj,
                    # initialization
                    attn_linear_init_mode=attn_linear_init_mode,
                    ffn_linear_init_mode=ffn_linear_init_mode,
                    norm_init_mode=norm_init_mode,
                ) for _ in range(ele_depth)]
            self_blocks.append(nn.ModuleList(stack_cuboid_blocks))
        self.self_blocks = nn.ModuleList(self_blocks)

        if block_cross_attn_patterns is not None:
            if isinstance(block_cross_attn_patterns, (tuple, list)):
                assert len(block_cross_attn_patterns) == self.num_blocks
            else:
                block_cross_attn_patterns = [block_cross_attn_patterns for _ in range(self.num_blocks)]

            block_cross_cuboid_hw = []
            block_cross_cuboid_strategy = []
            block_cross_shift_hw = []
            block_cross_n_temporal = []
            for idx, key in enumerate(block_cross_attn_patterns):
                if key == "last_frame_dst":
                    cuboid_hw = None
                    shift_hw = None
                    strategy = None
                    n_temporal = None
                else:
                    # func = CuboidCrossAttentionPatterns.get(key)
                    # cuboid_hw, shift_hw, strategy, n_temporal = func(mem_shapes[idx])
                    cuboid_hw, shift_hw, strategy, n_temporal = CuboidCrossAttentionPatterns.build(
                        dict(type=key, mem_shape=mem_shapes[idx])
                    )
                block_cross_cuboid_hw.append(cuboid_hw)
                block_cross_cuboid_strategy.append(strategy)
                block_cross_shift_hw.append(shift_hw)
                block_cross_n_temporal.append(n_temporal)
        else:
            if not isinstance(block_cross_cuboid_hw[0][0], (list, tuple)):
                block_cross_cuboid_hw = [block_cross_cuboid_hw for _ in range(self.num_blocks)]
            else:
                assert len(block_cross_cuboid_hw) == self.num_blocks, \
                    f'Incorrect input format! Received block_cross_cuboid_hw={block_cross_cuboid_hw}'

            if not isinstance(block_cross_cuboid_strategy[0][0], (list, tuple)):
                block_cross_cuboid_strategy = [block_cross_cuboid_strategy for _ in range(self.num_blocks)]
            else:
                assert len(block_cross_cuboid_strategy) == self.num_blocks, \
                    f'Incorrect input format! Received block_cross_cuboid_strategy={block_cross_cuboid_strategy}'

            if not isinstance(block_cross_shift_hw[0][0], (list, tuple)):
                block_cross_shift_hw = [block_cross_shift_hw for _ in range(self.num_blocks)]
            else:
                assert len(block_cross_shift_hw) == self.num_blocks, \
                    f'Incorrect input format! Received block_cross_shift_hw={block_cross_shift_hw}'
            if not isinstance(block_cross_n_temporal[0], (list, tuple)):
                block_cross_n_temporal = [block_cross_n_temporal for _ in range(self.num_blocks)]
            else:
                assert len(block_cross_n_temporal) == self.num_blocks, \
                    f'Incorrect input format! Received block_cross_n_temporal={block_cross_n_temporal}'
        self.cross_blocks = nn.ModuleList()
        for i in range(self.cross_start, self.num_blocks):
            cross_block = nn.ModuleList(
                [StackCuboidCrossAttentionBlock(
                    dim=self.mem_shapes[i][-1],
                    num_heads=num_heads,
                    block_cuboid_hw=block_cross_cuboid_hw[i],
                    block_strategy=block_cross_cuboid_strategy[i],
                    block_shift_hw=block_cross_shift_hw[i],
                    block_n_temporal=block_cross_n_temporal[i],
                    cross_last_n_frames=cross_last_n_frames,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    ffn_drop=ffn_drop,
                    gated_ffn=gated_ffn,
                    norm_layer=norm_layer,
                    use_inter_ffn=use_inter_ffn,
                    activation=ffn_activation,
                    max_temporal_relative=max_temporal_relative,
                    padding_type=padding_type,
                    use_global_vector=use_cross_global,
                    separate_global_qkv=separate_global_qkv,
                    global_dim_ratio=global_dim_ratio,
                    checkpoint_level=checkpoint_level,
                    use_relative_pos=use_relative_pos,
                    # initialization
                    attn_linear_init_mode=attn_linear_init_mode,
                    ffn_linear_init_mode=ffn_linear_init_mode,
                    norm_init_mode=norm_init_mode,
                ) for _ in range(depth[i])])
            self.cross_blocks.append(cross_block)

        # Construct upsampling layers
        if self.num_blocks > 1:
            if self.upsample_type == "upsample":
                self.upsample_layers = nn.ModuleList([
                    Upsample3DLayer(
                        dim=self.mem_shapes[i + 1][-1],
                        out_dim=self.mem_shapes[i][-1],
                        target_size=(target_temporal_length,) + self.mem_shapes[i][1:3],
                        kernel_size=upsample_kernel_size,
                        temporal_upsample=False,
                        conv_init_mode=conv_init_mode,
                    )
                    for i in range(self.num_blocks - 1)])
            else:
                raise NotImplementedError
            if self.hierarchical_pos_embed:
                self.hierarchical_pos_embed_l = nn.ModuleList([
                    PosEmbed(embed_dim=self.mem_shapes[i][-1], typ=pos_embed_type,
                             maxT=target_temporal_length, maxH=self.mem_shapes[i][1], maxW=self.mem_shapes[i][2])
                    for i in range(self.num_blocks - 1)])

        self.reset_parameters()

    def reset_parameters(self):
        for ms in self.self_blocks:
            for m in ms:
                m.reset_parameters()
        for ms in self.cross_blocks:
            for m in ms:
                m.reset_parameters()
        if self.num_blocks > 1:
            for m in self.upsample_layers:
                m.reset_parameters()
        if self.hierarchical_pos_embed:
            for m in self.hierarchical_pos_embed_l:
                m.reset_parameters()

    def forward(self, x, mem_l, mem_global_vector_l=None):
        """

        Parameters
        ----------
        x
            Shape (B, T_top, H_top, W_top, C)
        mem_l
            A list of memory tensors

        Returns
        -------
        out
        """
        B, T_top, H_top, W_top, C = x.shape
        assert T_top == self.target_temporal_length
        assert (H_top, W_top) == (self.mem_shapes[-1][1], self.mem_shapes[-1][2])
        for i in range(self.num_blocks - 1, -1, -1):
            mem_global_vector = None if mem_global_vector_l is None else mem_global_vector_l[i]
            if not self.use_first_self_attn and i == self.num_blocks - 1:
                # For the top block, we won't use the self attention layer and will directly use the cross attention layer.
                if i >= self.cross_start:
                    x = self.cross_blocks[i - self.cross_start][0](x, mem_l[i], mem_global_vector)
                for idx in range(self.depth[i] - 1):
                    if self.use_self_global:  # in this case `mem_global_vector` is guaranteed to be not None
                        if self.self_update_global:
                            x, mem_global_vector = self.self_blocks[i][idx](x, mem_global_vector)
                        else:
                            x, _ = self.self_blocks[i][idx](x, mem_global_vector)
                    else:
                        x = self.self_blocks[i][idx](x)
                    if i >= self.cross_start:
                        x = self.cross_blocks[i - self.cross_start][idx + 1](x, mem_l[i], mem_global_vector)
            else:
                for idx in range(self.depth[i]):
                    if self.use_self_global:
                        if self.self_update_global:
                            x, mem_global_vector = self.self_blocks[i][idx](x, mem_global_vector)
                        else:
                            x, _ = self.self_blocks[i][idx](x, mem_global_vector)
                    else:
                        x = self.self_blocks[i][idx](x)
                    if i >= self.cross_start:
                        x = self.cross_blocks[i - self.cross_start][idx](x, mem_l[i], mem_global_vector)
            # Upsample
            if i > 0:
                x = self.upsample_layers[i - 1](x)
                if self.hierarchical_pos_embed:
                    x = self.hierarchical_pos_embed_l[i - 1](x)
        return x

class InitialEncoder(nn.Module):
    def __init__(self,
                 dim,
                 out_dim,
                 downsample_scale: Union[int, Sequence[int]],
                 num_conv_layers=2,
                 activation='leaky',
                 padding_type='nearest',
                 conv_init_mode="0",
                 linear_init_mode="0",
                 norm_init_mode="0",
                 ):
        super(InitialEncoder, self).__init__()

        self.num_conv_layers = num_conv_layers
        self.conv_init_mode = conv_init_mode
        self.linear_init_mode = linear_init_mode
        self.norm_init_mode = norm_init_mode

        conv_block = []
        for i in range(num_conv_layers):
            if i == 0:
                conv_block.append(nn.Conv2d(kernel_size=(3, 3), padding=(1, 1),
                                            in_channels=dim, out_channels=out_dim))
                conv_block.append(nn.GroupNorm(16, out_dim))
                conv_block.append(get_activation(activation))
            else:
                conv_block.append(nn.Conv2d(kernel_size=(3, 3), padding=(1, 1),
                                            in_channels=out_dim, out_channels=out_dim))
                conv_block.append(nn.GroupNorm(16, out_dim))
                conv_block.append(get_activation(activation))

        self.conv_block = nn.Sequential(*conv_block)
        if isinstance(downsample_scale, int):
            patch_merge_downsample = (1, downsample_scale, downsample_scale)
        elif len(downsample_scale) == 2:
            patch_merge_downsample = (1, *downsample_scale)
        elif len(downsample_scale) == 3:
            patch_merge_downsample = tuple(downsample_scale)
        else:
            raise NotImplementedError(f"downsample_scale {downsample_scale} format not supported!")
        self.patch_merge = PatchMerging3D(
            dim=out_dim, out_dim=out_dim,
            padding_type=padding_type,
            downsample=patch_merge_downsample,
            linear_init_mode=linear_init_mode,
            norm_init_mode=norm_init_mode)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            apply_initialization(m,
                                 conv_mode=self.conv_init_mode,
                                 linear_mode=self.linear_init_mode,
                                 norm_mode=self.norm_init_mode)

    def forward(self, x):
        """

        x --> [K x Conv2D] --> PatchMerge

        Parameters
        ----------
        x
            Shape (B, T, H, W, C)

        Returns
        -------
        out
            Shape (B, T, H_new, W_new, C_out)
        """
        B, T, H, W, C = x.shape
        if self.num_conv_layers > 0:
            x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
            x = self.conv_block(x).permute(0, 2, 3, 1)  # (B * T, H, W, C_new)
            x = self.patch_merge(x.reshape(B, T, H, W, -1))
        else:
            x = self.patch_merge(x)
        return x

class FinalDecoder(nn.Module):

    def __init__(self,
                 target_thw,
                 dim,
                 num_conv_layers=2,
                 activation='leaky',
                 conv_init_mode="0",
                 linear_init_mode="0",
                 norm_init_mode="0",
                 ):
        super(FinalDecoder, self).__init__()
        # self.target_thw = target_thw
        # self.dim = dim
        self.num_conv_layers = num_conv_layers
        self.conv_init_mode = conv_init_mode
        self.linear_init_mode = linear_init_mode
        self.norm_init_mode = norm_init_mode

        conv_block = []
        for i in range(num_conv_layers):
            conv_block.append(nn.Conv2d(kernel_size=(3, 3), padding=(1, 1), in_channels=dim, out_channels=dim))
            conv_block.append(nn.GroupNorm(16, dim))
            conv_block.append(get_activation(activation))
        self.conv_block = nn.Sequential(*conv_block)
        self.upsample = Upsample3DLayer(
            dim=dim, out_dim=dim,
            target_size=target_thw, kernel_size=3,
            conv_init_mode=conv_init_mode)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            apply_initialization(m,
                                 conv_mode=self.conv_init_mode,
                                 linear_mode=self.linear_init_mode,
                                 norm_mode=self.norm_init_mode)

    def forward(self, x):
        """

        x --> Upsample --> [K x Conv2D]

        Parameters
        ----------
        x
            Shape (B, T, H, W, C)

        Returns
        -------
        out
            Shape (B, T, H_new, W_new, C)
        """
        x = self.upsample(x)
        if self.num_conv_layers > 0:
            B, T, H, W, C = x.shape
            x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
            x = self.conv_block(x).permute(0, 2, 3, 1).reshape(B, T, H, W, -1)
        return x

class InitialStackPatchMergingEncoder(nn.Module):

    def __init__(self,
                 num_merge: int,
                 in_dim,
                 out_dim_list,
                 downsample_scale_list,
                 num_conv_per_merge_list=None,
                 activation='leaky',
                 padding_type='nearest',
                 conv_init_mode="0",
                 linear_init_mode="0",
                 norm_init_mode="0",
                 ):
        super(InitialStackPatchMergingEncoder, self).__init__()

        self.conv_init_mode = conv_init_mode
        self.linear_init_mode = linear_init_mode
        self.norm_init_mode = norm_init_mode

        # self.num_merge = num_merge
        # self.in_dim = in_dim
        self.out_dim_list = out_dim_list[:num_merge]
        self.downsample_scale_list = downsample_scale_list[:num_merge]
        self.num_conv_per_merge_list = num_conv_per_merge_list
        self.num_group_list = [max(1, out_dim // 4) for out_dim in self.out_dim_list]

        self.conv_block_list = nn.ModuleList()
        self.patch_merge_list = nn.ModuleList()
        for i in range(num_merge):
            if i == 0:
                in_dim = in_dim
            else:
                in_dim = self.out_dim_list[i - 1]
            out_dim = self.out_dim_list[i]
            downsample_scale = self.downsample_scale_list[i]

            conv_block = []
            for j in range(self.num_conv_per_merge_list[i]):
                if j == 0:
                    conv_in_dim = in_dim
                else:
                    conv_in_dim = out_dim
                conv_block.append(nn.Conv2d(kernel_size=(3, 3), padding=(1, 1),
                                            in_channels=conv_in_dim, out_channels=out_dim))
                conv_block.append(nn.GroupNorm(self.num_group_list[i], out_dim))
                conv_block.append(get_activation(activation))

            conv_block = nn.Sequential(*conv_block)
            self.conv_block_list.append(conv_block)
            patch_merge = PatchMerging3D(
                dim=out_dim, out_dim=out_dim,
                padding_type=padding_type,
                downsample=(1, downsample_scale, downsample_scale),
                linear_init_mode=linear_init_mode,
                norm_init_mode=norm_init_mode)
            self.patch_merge_list.append(patch_merge)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            apply_initialization(m,
                                 conv_mode=self.conv_init_mode,
                                 linear_mode=self.linear_init_mode,
                                 norm_mode=self.norm_init_mode)

    def get_out_shape_list(self, input_shape):
        """
        T, H, W, C
        """
        out_shape_list = []
        for patch_merge in self.patch_merge_list:
            input_shape = patch_merge.get_out_shape(input_shape)
            out_shape_list.append(input_shape)
        return out_shape_list

    def forward(self, x):
        """

        x --> [K x Conv2D] --> PatchMerge --> ... --> [K x Conv2D] --> PatchMerge

        Parameters
        ----------
        x
            Shape (B, T, H, W, C)

        Returns
        -------
        out
            Shape (B, T, H_new, W_new, C_out)
        """
        for i, (conv_block, patch_merge) in \
                enumerate(zip(self.conv_block_list, self.patch_merge_list)):
            B, T, H, W, C = x.shape
            if self.num_conv_per_merge_list[i] > 0:
                x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
                x = conv_block(x).permute(0, 2, 3, 1).reshape(B, T, H, W, -1)
            x = patch_merge(x)
        return x

class FinalStackUpsamplingDecoder(nn.Module):

    def __init__(self,
                 target_shape_list,
                 in_dim,
                 num_conv_per_up_list=None,
                 activation='leaky',
                 conv_init_mode="0",
                 linear_init_mode="0",
                 norm_init_mode="0",
                 ):
        """
        Parameters
        ----------
        target_shape_list:
            list of (T, H ,W ,C)
        """
        super(FinalStackUpsamplingDecoder, self).__init__()
        self.conv_init_mode = conv_init_mode
        self.linear_init_mode = linear_init_mode
        self.norm_init_mode = norm_init_mode

        # self.target_shape_list = target_shape_list
        self.out_dim_list = [target_shape[-1] for target_shape in target_shape_list]
        self.num_upsample = len(target_shape_list)
        # self.in_dim = in_dim
        self.num_conv_per_up_list = num_conv_per_up_list
        self.num_group_list = [max(1, out_dim // 4) for out_dim in self.out_dim_list]

        self.conv_block_list = nn.ModuleList()
        self.upsample_list = nn.ModuleList()
        for i in range(self.num_upsample):
            if i == 0:
                in_dim = in_dim
            else:
                in_dim = self.out_dim_list[i - 1]
            out_dim = self.out_dim_list[i]

            upsample = Upsample3DLayer(
                dim=in_dim, out_dim=in_dim,
                target_size=target_shape_list[i][:-1], kernel_size=3,
                conv_init_mode=conv_init_mode)
            self.upsample_list.append(upsample)
            conv_block = []
            for j in range(num_conv_per_up_list[i]):
                if j == 0:
                    conv_in_dim = in_dim
                else:
                    conv_in_dim = out_dim
                conv_block.append(nn.Conv2d(kernel_size=(3, 3), padding=(1, 1),
                                            in_channels=conv_in_dim, out_channels=out_dim))
                conv_block.append(nn.GroupNorm(self.num_group_list[i], out_dim))
                conv_block.append(get_activation(activation))
            conv_block = nn.Sequential(*conv_block)
            self.conv_block_list.append(conv_block)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            apply_initialization(m,
                                 conv_mode=self.conv_init_mode,
                                 linear_mode=self.linear_init_mode,
                                 norm_mode=self.norm_init_mode)

    @staticmethod
    def get_init_params(enc_input_shape, enc_out_shape_list, large_channel=False):
        dec_target_shape_list = list(enc_out_shape_list[:-1])[::-1] + [tuple(enc_input_shape), ]
        if large_channel:
            dec_target_shape_list_large_channel = []
            for i, enc_out_shape in enumerate(enc_out_shape_list[::-1]):
                dec_target_shape_large_channel = list(dec_target_shape_list[i])
                dec_target_shape_large_channel[-1] = enc_out_shape[-1]
                dec_target_shape_list_large_channel.append(tuple(dec_target_shape_large_channel))
            dec_target_shape_list = dec_target_shape_list_large_channel
        dec_in_dim = enc_out_shape_list[-1][-1]
        return dec_target_shape_list, dec_in_dim

    def forward(self, x):
        """

        x --> Upsample --> [K x Conv2D] --> ... --> Upsample --> [K x Conv2D]

        Parameters
        ----------
        x
            Shape (B, T, H, W, C)

        Returns
        -------
        out
            Shape (B, T, H_new, W_new, C)
        """
        for i, (conv_block, upsample) in \
                enumerate(zip(self.conv_block_list, self.upsample_list)):
            x = upsample(x)
            if self.num_conv_per_up_list[i] > 0:
                B, T, H, W, C = x.shape
                x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
                x = conv_block(x).permute(0, 2, 3, 1).reshape(B, T, H, W, -1)
        return x





@MODELS.register_module()
class CuboidTransformerModel(nn.Module):
    """Cuboid Transformer for spatiotemporal forecasting

    We adopt the Non-autoregressive encoder-decoder architecture.
    The decoder takes the multi-scale memory output from the encoder.

    The initial downsampling / upsampling layers will be
    Downsampling: [K x Conv2D --> PatchMerge]
    Upsampling: [Nearest Interpolation-based Upsample --> K x Conv2D]

    x --> downsample (optional) ---> (+pos_embed) ---> enc --> mem_l         initial_z (+pos_embed) ---> FC
                                                     |            |
                                                     |------------|
                                                           |
                                                           |
             y <--- upsample (optional) <--- dec <----------

    """
    def __init__(self,
                 input_shape,
                 target_shape,
                 base_units=128,
                 block_units=None,
                 scale_alpha=1.0,
                 num_heads=4,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 ffn_drop=0.0,
                 # inter-attn downsample/upsample
                 downsample=2,
                 downsample_type='patch_merge',
                 upsample_type="upsample",
                 upsample_kernel_size=3,
                 # encoder
                 enc_depth=[4, 4, 4],
                 enc_attn_patterns=None,
                 enc_cuboid_size=[(4, 4, 4), (4, 4, 4)],
                 enc_cuboid_strategy=[('l', 'l', 'l'), ('d', 'd', 'd')],
                 enc_shift_size=[(0, 0, 0), (0, 0, 0)],
                 enc_use_inter_ffn=True,
                 # decoder
                 dec_depth=[2, 2],
                 dec_cross_start=0,
                 dec_self_attn_patterns=None,
                 dec_self_cuboid_size=[(4, 4, 4), (4, 4, 4)],
                 dec_self_cuboid_strategy=[('l', 'l', 'l'), ('d', 'd', 'd')],
                 dec_self_shift_size=[(1, 1, 1), (0, 0, 0)],
                 dec_cross_attn_patterns=None,
                 dec_cross_cuboid_hw=[(4, 4), (4, 4)],
                 dec_cross_cuboid_strategy=[('l', 'l', 'l'), ('d', 'l', 'l')],
                 dec_cross_shift_hw=[(0, 0), (0, 0)],
                 dec_cross_n_temporal=[1, 2],
                 dec_cross_last_n_frames=None,
                 dec_use_inter_ffn=True,
                 dec_hierarchical_pos_embed=False,
                 # global vectors
                 num_global_vectors=4,
                 use_dec_self_global=True,
                 dec_self_update_global=True,
                 use_dec_cross_global=True,
                 use_global_vector_ffn=True,
                 use_global_self_attn=False,
                 separate_global_qkv=False,
                 global_dim_ratio=1,
                 z_init_method='nearest_interp',
                 # # initial downsample and final upsample
                 initial_downsample_type="conv",
                 initial_downsample_activation="leaky",
                 # initial_downsample_type=="conv"
                 initial_downsample_scale=1,
                 initial_downsample_conv_layers=2,
                 final_upsample_conv_layers=2,
                 # initial_downsample_type == "stack_conv"
                 initial_downsample_stack_conv_num_layers=1,
                 initial_downsample_stack_conv_dim_list=None,
                 initial_downsample_stack_conv_downscale_list=[1, ],
                 initial_downsample_stack_conv_num_conv_list=[2, ],
                 # # end of initial downsample and final upsample
                 ffn_activation='leaky',
                 gated_ffn=False,
                 norm_layer='layer_norm',
                 padding_type='ignore',
                 pos_embed_type='t+hw',
                 checkpoint_level=True,
                 use_relative_pos=True,
                 self_attn_use_final_proj=True,
                 dec_use_first_self_attn=False,
                 # initialization
                 attn_linear_init_mode="0",
                 ffn_linear_init_mode="0",
                 conv_init_mode="0",
                 down_up_linear_init_mode="0",
                 norm_init_mode="0",
                 ):
        """

        Parameters
        ----------
        input_shape
            Shape of the input tensor. It will be (T, H, W, C_in)
        target_shape
            Shape of the input tensor. It will be (T_out, H, W, C_out)
        base_units
            The base units
        z_init_method
            How the initial input to the decoder is initialized
        """
        super(CuboidTransformerModel, self).__init__()
        # initialization mode
        # self.attn_linear_init_mode = attn_linear_init_mode
        # self.ffn_linear_init_mode = ffn_linear_init_mode
        self.conv_init_mode = conv_init_mode
        self.down_up_linear_init_mode = down_up_linear_init_mode
        self.norm_init_mode = norm_init_mode

        assert len(enc_depth) == len(dec_depth)
        self.base_units = base_units
        self.num_global_vectors = num_global_vectors
        if global_dim_ratio != 1:
            assert separate_global_qkv == True, \
                f"Setting global_dim_ratio != 1 requires separate_global_qkv == True."
        self.global_dim_ratio = global_dim_ratio
        self.z_init_method = z_init_method
        assert self.z_init_method in ['zeros', 'nearest_interp', 'last', 'mean']

        self.input_shape = input_shape
        self.target_shape = target_shape
        T_in, H_in, W_in, C_in = input_shape
        T_out, H_out, W_out, C_out = target_shape
        assert H_in == H_out and W_in == W_out

        if self.num_global_vectors > 0:
            self.init_global_vectors = nn.Parameter(
                torch.zeros((self.num_global_vectors, global_dim_ratio*base_units)))

        new_input_shape = self.get_initial_encoder_final_decoder(
            initial_downsample_scale=initial_downsample_scale,
            initial_downsample_type=initial_downsample_type,
            activation=initial_downsample_activation,
            # initial_downsample_type=="conv"
            initial_downsample_conv_layers=initial_downsample_conv_layers,
            final_upsample_conv_layers=final_upsample_conv_layers,
            padding_type=padding_type,
            # initial_downsample_type == "stack_conv"
            initial_downsample_stack_conv_num_layers=initial_downsample_stack_conv_num_layers,
            initial_downsample_stack_conv_dim_list=initial_downsample_stack_conv_dim_list,
            initial_downsample_stack_conv_downscale_list=initial_downsample_stack_conv_downscale_list,
            initial_downsample_stack_conv_num_conv_list=initial_downsample_stack_conv_num_conv_list,
        )
        T_in, H_in, W_in, _ = new_input_shape

        self.encoder = CuboidTransformerEncoder(
            input_shape=(T_in, H_in, W_in, base_units),
            base_units=base_units,
            block_units=block_units,
            scale_alpha=scale_alpha,
            depth=enc_depth,
            downsample=downsample,
            downsample_type=downsample_type,
            block_attn_patterns=enc_attn_patterns,
            block_cuboid_size=enc_cuboid_size,
            block_strategy=enc_cuboid_strategy,
            block_shift_size=enc_shift_size,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            ffn_drop=ffn_drop,
            gated_ffn=gated_ffn,
            ffn_activation=ffn_activation,
            norm_layer=norm_layer,
            use_inter_ffn=enc_use_inter_ffn,
            padding_type=padding_type,
            use_global_vector=num_global_vectors > 0,
            use_global_vector_ffn=use_global_vector_ffn,
            use_global_self_attn=use_global_self_attn,
            separate_global_qkv=separate_global_qkv,
            global_dim_ratio=global_dim_ratio,
            checkpoint_level=checkpoint_level,
            use_relative_pos=use_relative_pos,
            self_attn_use_final_proj=self_attn_use_final_proj,
            # initialization
            attn_linear_init_mode=attn_linear_init_mode,
            ffn_linear_init_mode=ffn_linear_init_mode,
            conv_init_mode=conv_init_mode,
            down_linear_init_mode=down_up_linear_init_mode,
            norm_init_mode=norm_init_mode,
        )
        self.enc_pos_embed = PosEmbed(
            embed_dim=base_units, typ=pos_embed_type,
            maxH=H_in, maxW=W_in, maxT=T_in)
        mem_shapes = self.encoder.get_mem_shapes()

        self.z_proj = nn.Linear(mem_shapes[-1][-1], mem_shapes[-1][-1])
        self.dec_pos_embed = PosEmbed(
            embed_dim=mem_shapes[-1][-1], typ=pos_embed_type,
            maxT=T_out, maxH=mem_shapes[-1][1], maxW=mem_shapes[-1][2])
        self.decoder = CuboidTransformerDecoder(
            target_temporal_length=T_out,
            mem_shapes=mem_shapes,
            cross_start=dec_cross_start,
            depth=dec_depth,
            upsample_type=upsample_type,
            block_self_attn_patterns=dec_self_attn_patterns,
            block_self_cuboid_size=dec_self_cuboid_size,
            block_self_shift_size=dec_self_shift_size,
            block_self_cuboid_strategy=dec_self_cuboid_strategy,
            block_cross_attn_patterns=dec_cross_attn_patterns,
            block_cross_cuboid_hw=dec_cross_cuboid_hw,
            block_cross_shift_hw=dec_cross_shift_hw,
            block_cross_cuboid_strategy=dec_cross_cuboid_strategy,
            block_cross_n_temporal=dec_cross_n_temporal,
            cross_last_n_frames=dec_cross_last_n_frames,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            ffn_drop=ffn_drop,
            upsample_kernel_size=upsample_kernel_size,
            ffn_activation=ffn_activation,
            gated_ffn=gated_ffn,
            norm_layer=norm_layer,
            use_inter_ffn=dec_use_inter_ffn,
            max_temporal_relative=T_in + T_out,
            padding_type=padding_type,
            hierarchical_pos_embed=dec_hierarchical_pos_embed,
            pos_embed_type=pos_embed_type,
            use_self_global=(num_global_vectors > 0) and use_dec_self_global,
            self_update_global=dec_self_update_global,
            use_cross_global=(num_global_vectors > 0) and use_dec_cross_global,
            use_global_vector_ffn=use_global_vector_ffn,
            use_global_self_attn=use_global_self_attn,
            separate_global_qkv=separate_global_qkv,
            global_dim_ratio=global_dim_ratio,
            checkpoint_level=checkpoint_level,
            use_relative_pos=use_relative_pos,
            self_attn_use_final_proj=self_attn_use_final_proj,
            use_first_self_attn=dec_use_first_self_attn,
            # initialization
            attn_linear_init_mode=attn_linear_init_mode,
            ffn_linear_init_mode=ffn_linear_init_mode,
            conv_init_mode=conv_init_mode,
            up_linear_init_mode=down_up_linear_init_mode,
            norm_init_mode=norm_init_mode,
        )
        self.reset_parameters()
        

    def get_initial_encoder_final_decoder(
            self,
            initial_downsample_type,
            activation,
            # initial_downsample_type=="conv"
            initial_downsample_scale,
            initial_downsample_conv_layers,
            final_upsample_conv_layers,
            padding_type,
            # initial_downsample_type == "stack_conv"
            initial_downsample_stack_conv_num_layers,
            initial_downsample_stack_conv_dim_list,
            initial_downsample_stack_conv_downscale_list,
            initial_downsample_stack_conv_num_conv_list,
        ):
        T_in, H_in, W_in, C_in = self.input_shape
        T_out, H_out, W_out, C_out = self.target_shape
        # Construct the initial upsampling / downsampling layers
        self.initial_downsample_type = initial_downsample_type
        if self.initial_downsample_type == "conv":
            if isinstance(initial_downsample_scale, int):
                initial_downsample_scale = (1, initial_downsample_scale, initial_downsample_scale)
            elif len(initial_downsample_scale) == 2:
                initial_downsample_scale = (1, *initial_downsample_scale)
            elif len(initial_downsample_scale) == 3:
                initial_downsample_scale = tuple(initial_downsample_scale)
            else:
                raise NotImplementedError(f"initial_downsample_scale {initial_downsample_scale} format not supported!")
            # if any(ele > 1 for ele in initial_downsample_scale):
            self.initial_encoder = InitialEncoder(dim=C_in,
                                                  out_dim=self.base_units,
                                                  downsample_scale=initial_downsample_scale,
                                                  num_conv_layers=initial_downsample_conv_layers,
                                                  padding_type=padding_type,
                                                  activation=activation,
                                                  conv_init_mode=self.conv_init_mode,
                                                  linear_init_mode=self.down_up_linear_init_mode,
                                                  norm_init_mode=self.norm_init_mode)
            self.final_decoder = FinalDecoder(dim=self.base_units,
                                              target_thw=(T_out, H_out, W_out),
                                              num_conv_layers=final_upsample_conv_layers,
                                              activation=activation,
                                              conv_init_mode=self.conv_init_mode,
                                              linear_init_mode=self.down_up_linear_init_mode,
                                              norm_init_mode=self.norm_init_mode)
            new_input_shape = self.initial_encoder.patch_merge.get_out_shape(self.input_shape)
            self.dec_final_proj = nn.Linear(self.base_units, C_out)
            # else:
            #     self.initial_encoder = nn.Linear(C_in, self.base_units)
            #     self.final_decoder = nn.Identity()
            #     self.dec_final_proj = nn.Linear(self.base_units, C_out)
            #     new_input_shape = self.input_shape

        elif self.initial_downsample_type == "stack_conv":
            if initial_downsample_stack_conv_dim_list is None:
                initial_downsample_stack_conv_dim_list = [self.base_units, ] * initial_downsample_stack_conv_num_layers
            self.initial_encoder = InitialStackPatchMergingEncoder(
                num_merge=initial_downsample_stack_conv_num_layers,
                in_dim=C_in,
                out_dim_list=initial_downsample_stack_conv_dim_list,
                downsample_scale_list=initial_downsample_stack_conv_downscale_list,
                num_conv_per_merge_list=initial_downsample_stack_conv_num_conv_list,
                padding_type=padding_type,
                activation=activation,
                conv_init_mode=self.conv_init_mode,
                linear_init_mode=self.down_up_linear_init_mode,
                norm_init_mode=self.norm_init_mode)
            # use `self.target_shape` to get correct T_out
            initial_encoder_out_shape_list = self.initial_encoder.get_out_shape_list(self.target_shape)
            dec_target_shape_list, dec_in_dim = \
                FinalStackUpsamplingDecoder.get_init_params(
                    enc_input_shape=self.target_shape,
                    enc_out_shape_list=initial_encoder_out_shape_list,
                    large_channel=True)
            self.final_decoder = FinalStackUpsamplingDecoder(
                target_shape_list=dec_target_shape_list,
                in_dim=dec_in_dim,
                num_conv_per_up_list=initial_downsample_stack_conv_num_conv_list[::-1],
                activation=activation,
                conv_init_mode=self.conv_init_mode,
                linear_init_mode=self.down_up_linear_init_mode,
                norm_init_mode=self.norm_init_mode)
            self.dec_final_proj = nn.Linear(dec_target_shape_list[-1][-1], C_out)
            new_input_shape = self.initial_encoder.get_out_shape_list(self.input_shape)[-1]
        else:
            raise NotImplementedError
        # self.input_shape_after_initial_downsample = new_input_shape
        # T_in, H_in, W_in, _ = new_input_shape

        return new_input_shape

    def reset_parameters(self):
        if self.num_global_vectors > 0:
            nn.init.trunc_normal_(self.init_global_vectors, std=.02)
        if hasattr(self.initial_encoder, "reset_parameters"):
            self.initial_encoder.reset_parameters()
        else:
            apply_initialization(self.initial_encoder,
                                 conv_mode=self.conv_init_mode,
                                 linear_mode=self.down_up_linear_init_mode,
                                 norm_mode=self.norm_init_mode)
        if hasattr(self.final_decoder, "reset_parameters"):
            self.final_decoder.reset_parameters()
        else:
            apply_initialization(self.final_decoder,
                                 conv_mode=self.conv_init_mode,
                                 linear_mode=self.down_up_linear_init_mode,
                                 norm_mode=self.norm_init_mode)
        apply_initialization(self.dec_final_proj,
                             linear_mode=self.down_up_linear_init_mode)
        self.encoder.reset_parameters()
        self.enc_pos_embed.reset_parameters()
        self.decoder.reset_parameters()
        self.dec_pos_embed.reset_parameters()
        apply_initialization(self.z_proj,
                             linear_mode="0")

    def get_initial_z(self, final_mem, T_out):
        B = final_mem.shape[0]
        if self.z_init_method == 'zeros':
            z_shape = (1, T_out) + final_mem.shape[2:]
            initial_z = torch.zeros(z_shape, dtype=final_mem.dtype, device=final_mem.device)
            initial_z = self.z_proj(self.dec_pos_embed(initial_z)).expand(B, -1, -1, -1, -1)
        elif self.z_init_method == 'nearest_interp':
            # final_mem will have shape (B, T, H, W, C)
            initial_z = F.interpolate(final_mem.permute(0, 4, 1, 2, 3),
                                      size=(T_out, final_mem.shape[2], final_mem.shape[3])).permute(0, 2, 3, 4, 1)
            initial_z = self.z_proj(initial_z)
        elif self.z_init_method == 'last':
            initial_z = torch.broadcast_to(final_mem[:, -1:, :, :, :], (B, T_out) + final_mem.shape[2:])
            initial_z = self.z_proj(initial_z)
        elif self.z_init_method == 'mean':
            initial_z = torch.broadcast_to(final_mem.mean(axis=1, keepdims=True),
                                           (B, T_out) + final_mem.shape[2:])
            initial_z = self.z_proj(initial_z)
        else:
            raise NotImplementedError
        return initial_z

    def forward(self, x, verbose=False):
        """

        Parameters
        ----------
        x
            Shape (B, T, H, W, C)
        verbos
            if True, print intermediate shapes
        Returns
        -------
        out
            The output Shape (B, T_out, H, W, C_out)
        """
        B, _, _, _, _ = x.shape
        T_out = self.target_shape[0]
        x = self.initial_encoder(x)
        x = self.enc_pos_embed(x)
        if self.num_global_vectors > 0:
            init_global_vectors = self.init_global_vectors\
                .expand(B, self.num_global_vectors, self.global_dim_ratio*self.base_units)
            mem_l, mem_global_vector_l = self.encoder(x, init_global_vectors)
            
        else:
            mem_l = self.encoder(x)
        if verbose:
            for i, mem in enumerate(mem_l):
                print(f"mem[{i}].shape = {mem.shape}")
        initial_z = self.get_initial_z(final_mem=mem_l[-1],
                                       T_out=T_out)
        if self.num_global_vectors > 0:
            dec_out = self.decoder(initial_z, mem_l, mem_global_vector_l)
        else:
            dec_out = self.decoder(initial_z, mem_l)
        dec_out = self.final_decoder(dec_out)
        out = self.dec_final_proj(dec_out)
        return out
