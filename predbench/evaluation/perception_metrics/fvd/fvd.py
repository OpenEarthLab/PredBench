# modified from https://github.com/universome/fvd-comparison/blob/master/our_fvd.py

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import center_crop
import einops
from .i3d_pytorch import InceptionI3d



@torch.no_grad()
class FVDFeatureExtractor:
    '''
    videos_real and videos_fake has the shape of [n t c h w] 
    i3d_type must in ['i3d_jit', 'i3d_400', 'i3d_600']
    '''
    def __init__(self, i3d_type='i3d_400', use_gpu=False, resize_crop=False):
        self.use_gpu = use_gpu
        self.resize_crop = resize_crop
        self.i3d_type = i3d_type
        if self.i3d_type == 'i3d_jit':
            self.extractor = torch.jit.load("pre_download/i3d/i3d_torchscript.pt").eval()
        elif self.i3d_type == 'i3d_400':
            self.extractor = InceptionI3d(num_classes=400, in_channels=3).eval()
            self.extractor.load_state_dict(torch.load('pre_download/i3d/i3d_pretrained_400.pth'))
        elif self.i3d_type == 'i3d_600':
            self.extractor = InceptionI3d(num_classes=400, in_channels=3).eval()
            self.extractor.load_state_dict(torch.load('pre_download/i3d/i3d_pretrained_400.pth'))
        else:
            raise NotImplementedError("i3d_type must in ['i3d_jit', 'i3d_400', 'i3d_600']")
        if torch.cuda.is_available() and self.use_gpu:
            self.extractor = self.extractor.cuda()
            
    def __call__(self, videos_fake, videos_real):
        # videos_fake = torch.randn((16, 10, 3, 256, 512), dtype=videos_fake.dtype, device=videos_fake.device)
        # videos_real = torch.randn((16, 10, 3, 256, 512), dtype=videos_real.dtype, device=videos_real.device)
        videos_fake = einops.rearrange(
            self.bilinear_interpolation(videos_fake), 'n t c h w -> n c t h w'
        )
        videos_real = einops.rearrange(
            self.bilinear_interpolation(videos_real), 'n t c h w -> n c t h w'
        )
        if torch.cuda.is_available() and self.use_gpu:
            videos_fake, videos_real = videos_fake.cuda(), videos_real.cuda()
        else:
            videos_fake, videos_real = videos_fake.cpu(), videos_real.cpu()
        N, C, T, H, W = videos_fake.shape
        if T < 10:  # short video to long video
            videos_fake_ = torch.zeros((N, C, 10, H, W), dtype=videos_fake.dtype, device=videos_fake.device)
            videos_fake_[:, :, 0: T, ...] = videos_fake
            videos_real_ = torch.zeros((N, C, 10, H, W), dtype=videos_real.dtype, device=videos_real.device)
            videos_real_[:, :, 0: T, ...] = videos_real
        else:
            videos_fake_ = videos_fake
            videos_real_ = videos_real
        if C == 1:  # grey images to RGB images
            videos_fake_ = videos_fake_.repeat(1, 3, 1, 1, 1)
            videos_real_ = videos_real_.repeat(1, 3, 1, 1, 1)
        if self.i3d_type == 'i3d_jit':
            detector_kwargs = dict(rescale=False, resize=False, return_features=True)
            feats_fake = self.extractor(videos_fake_, **detector_kwargs).detach().cpu()
            feats_real = self.extractor(videos_real_, **detector_kwargs).detach().cpu()
        else:
            feats_fake = self.extractor(videos_fake_).detach().cpu()
            feats_real = self.extractor(videos_real_).detach().cpu()
        # print(feats_fake.shape, feats_real.shape)
        return torch.stack([feats_fake, feats_real], dim=0)
        
    def bilinear_interpolation(self, videos):
        N, T, C, H, W = videos.shape
        def _resize_video(videos):
            # videos = videos.view(-1, C, H, W).contiguous()
            videos = einops.rearrange(videos, 'n t c h w -> (n t) c h w')
            videos = F.interpolate(videos, size=(224, 224), mode='bilinear', align_corners=False)
            videos = einops.rearrange(videos, '(n t) c h w -> n t c h w', n=N)
            # videos = videos.view(N, T, C, 224, 224)  
            return videos
        def _resize_crop_video(videos):
            videos = videos.view(-1, C, H, W).contiguous()
            if H<W:
                videos = F.interpolate(
                    videos, size=(224, int(W*224/H)), mode='bilinear', align_corners=False
                )
                videos = videos.view(N, T, C, 224, int(W*224/H))  
            else:   # W<=H
                videos = F.interpolate(
                    videos, size=(int(H*224/W), 224), mode='bilinear', align_corners=False
                )
                videos = videos.view(N, T, C, int(H*224/W), 224)  
            return center_crop(videos, (224, 224))
        if H == W and H < 224:
            return _resize_video(videos=videos)
        elif self.resize_crop:
            return _resize_crop_video(videos=videos)
        else: 
            return _resize_video(videos=videos)

def compute_FVD(feats_fake, feats_real):
    def compute_stats(feats):
        mu = np.mean(feats, axis=0)
        sigma = np.cov(feats, rowvar=False)
        return mu, sigma
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)
    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)
    fvd = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fvd)
