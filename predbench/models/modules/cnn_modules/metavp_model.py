from .metavp_modules import MetaVPEncoder, MetaVPDecoder, MidIncepNet, MidMetaNet
import einops
from torch.nn import Module

from predbench.registry import MODELS




@MODELS.register_module()
class MetaVPModel(Module):
    def __init__(self, 
        input_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gSTA',mlp_ratio=8., 
        drop=0.0, drop_path=0.0, spatio_kernel_enc=3, spatio_kernel_dec=3, act_inplace=True
    ):
        super().__init__()
        T, C, H, W = input_shape
        self.enc = MetaVPEncoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.dec = MetaVPDecoder(hid_S, C, N_S, spatio_kernel_dec, act_inplace=act_inplace)
        if model_type.lower() == 'incepu':
            self.midnet = MidIncepNet(T*hid_S, hid_T, N_T)
        else:
            self.midnet = MidMetaNet(
                T*hid_S, hid_T, N_T, input_resolution=(H, W), model_type=model_type.lower(),
                mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path
            )
    

    def forward(self, input):
        B, T, C, H, W = input.shape
        x = einops.rearrange(input, 'b t c h w -> (b t) c h w')
        
        embed, skip = self.enc(x)

        z = einops.rearrange(embed, '(b t) c h w -> b t c h w', b=B)
        hid = self.midnet(z)
        hid = einops.rearrange(hid, 'b t c h w -> (b t) c h w')

        pred = self.dec(hid, skip)
        pred = einops.rearrange(pred, '(b t) c h w -> b t c h w', b=B)
        return pred

