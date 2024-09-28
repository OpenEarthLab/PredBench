# modified from https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
#!/usr/bin/env python3
"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import os
import pathlib
import torch
import urllib

from scipy import linalg
import torch.nn.functional as F
from torchvision.transforms.functional import center_crop
import einops

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from .inception_v3 import InceptionV3


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('FID: fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return float(diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


@torch.no_grad()
class FIDFeatureExtractor:
    '''
    dims: (dims to block_index)
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    images_pred and images_true has the shape of [n c h w]
    '''
    def __init__(self, dims, use_gpu=False, resize_crop=False):
        self.use_gpu = use_gpu
        self.resize_crop = resize_crop
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.extractor = InceptionV3([block_idx]).eval()
        if torch.cuda.is_available() and self.use_gpu:
            self.extractor = self.extractor.cuda()
            
    def __call__(self, images_fake, images_real):
        # images_fake = torch.rand((64, 3, 512, 256), dtype=images_fake.dtype, device=images_fake.device)
        # images_real = torch.rand((64, 3, 512, 256), dtype=images_real.dtype, device=images_real.device)
        
        images_fake = self.bilinear_interpolation(images=images_fake)
        images_real = self.bilinear_interpolation(images=images_real)
        # print(images_fake.shape, images_real.shape)
        if torch.cuda.is_available() and self.use_gpu:
            images_fake, images_real = images_fake.cuda(), images_real.cuda()
        else:
            images_fake, images_real = images_fake.cpu(), images_real.cpu()
        if images_real.shape[-3] == 1:  # n c h w
            images_fake = images_fake.repeat(1, 3, 1, 1)
            images_real = images_real.repeat(1, 3, 1, 1)
        act_fake = self.extractor(images_fake)[0]   # b dims 1 1
        act_real = self.extractor(images_real)[0]   # b dims 1 1
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if act_fake.shape[2] != 1 or act_fake.shape[3] != 1:
            act_fake = F.adaptive_avg_pool2d(act_fake, output_size=(1, 1))
            act_real = F.adaptive_avg_pool2d(act_real, output_size=(1, 1))
        # print(act_fake.shape, act_real.shape)
        act_fake = einops.rearrange(act_fake.detach().cpu(), 'b d 1 1 -> b d')
        act_real = einops.rearrange(act_real.detach().cpu(), 'b d 1 1 -> b d')
        return torch.stack([act_fake, act_real], dim=0)
        
    
    def bilinear_interpolation(self, images):
        N, C, H, W = images.shape
        def _resize_image(images):
            return F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        def _resize_crop_image(images):
            if H<W:
                images = F.interpolate(
                    images, size=(299, int(W*299/H)), mode='bilinear', align_corners=False
                )
            else:   # W<=H
                images = F.interpolate(
                    images, size=(int(H*299/W), 299), mode='bilinear', align_corners=False
                )
            return center_crop(images, (299, 299))  
        if H == W and H < 299:
            return _resize_image(images=images)
        elif self.resize_crop:
            return _resize_crop_image(images=images)
        else: 
            return _resize_image(images=images)
        
        
        
def compute_FID(act_fake, act_real):
    def compute_stats(act):
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma
    mu_fake, sigma_fake = compute_stats(act_fake)
    mu_real, sigma_real = compute_stats(act_real)
    fid = calculate_frechet_distance(mu_fake, sigma_fake, mu_real, sigma_real)
    return fid












# def calculate_activations(images, model, batch_size=50, dims=2048, device=torch.device('cpu')):
#     """Calculates the activations of the pool_3 layer for all images.

#     Params:
#     -- images      : Tensor of images (n, 3, H, W), float values in [0, 1]
#     -- model       : Instance of inception model
#     -- batch_size  : Batch size of images for the model to process at once.
#                      Make sure that the number of samples is a multiple of
#                      the batch size, otherwise some samples are ignored. This
#                      behavior is retained to match the original FID score
#                      implementation.
#     -- dims        : Dimensionality of features returned by Inception
#     -- cuda        : If set to True, use GPU
    
#     Returns:
#     -- A numpy array of dimension (num images, dims) that contains the
#        activations of the given tensor when feeding inception with the
#        query tensor.
#     """
#     model.eval()

#     if batch_size > len(images):
#         print(('FID: Warning: batch size is bigger than the data size. '
#                'Setting batch size to data size'))
#         batch_size = len(images)

#     pred_arr = torch.empty((len(images), dims))

#     for i in tqdm(range(0, len(images), batch_size), leave=False, desc='InceptionV3'):
#         start = i
#         end = i + batch_size

#         # images = np.array([imread(str(f)).astype(np.float32)
#         #                    for f in files[start:end]])
#         # # Reshape to (n_images, 3, height, width)
#         # images = images.transpose((0, 3, 1, 2))
#         # images /= 255
#         # batch = torch.from_numpy(images).type(torch.FloatTensor)
#         batch = images[start:end].to(device)

#         pred = model(batch)[0]

#         # If model output is not scalar, apply global spatial average pooling.
#         # This happens if you choose a dimensionality not equal 2048.
#         if pred.size(2) != 1 or pred.size(3) != 1:
#             pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))

#         pred_arr[start:end] = pred.cpu().reshape(pred.size(0), -1)

#     return pred_arr


# def calculate_activation_statistics(
#     images, model, batch_size=50, dims=2048, device=torch.device('cpu')
# ):
#     """Calculation of the statistics used by the FID.
#     Params:
#     -- images      : Tensor of images (n, 3, H, W), float values in [0, 1]
#     -- model       : Instance of inception model
#     -- batch_size  : The images numpy array is split into batches with
#                      batch size batch_size. A reasonable batch size
#                      depends on the hardware.
#     -- dims        : Dimensionality of features returned by Inception
#     -- cuda        : If set to True, use GPU
#     -- verbose     : If set to True and parameter out_step is given, the
#                      number of calculated batches is reported.
#     Returns:
#     -- mu    : The mean over samples of the activations of the pool_3 layer of
#                the inception model.
#     -- sigma : The covariance matrix of the activations of the pool_3 layer of
#                the inception model.
#     """
#     act = calculate_activations(images, model, batch_size, dims, device).data.numpy()
#     mu = np.mean(act, axis=0)
#     sigma = np.cov(act, rowvar=False)
#     return mu, sigma


# def _compute_statistics_of_path_or_samples(path_or_samples, model, batch_size, dims, device):
#     if isinstance(path_or_samples, str):
#         assert path_or_samples.endswith('.npz'), "path is not .npz!"
#         f = np.load(path_or_samples)
#         m, s = f['mu'][:], f['sigma'][:]
#         f.close()
#     else:
#         assert isinstance(path_or_samples, torch.Tensor), "sample is not tensor!"
#         # path = pathlib.Path(path)
#         # files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
#         m, s = calculate_activation_statistics(path_or_samples, model, batch_size, dims, device)
#     return m, s


# def get_fid(path_or_samples1, path_or_samples2, device=torch.device('cuda'), batch_size=50, dims=2048):
#     block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
#     model = InceptionV3([block_idx]).to(device)
#     m1, s1 = _compute_statistics_of_path_or_samples(path_or_samples1, model, batch_size, dims, device)
#     m2, s2 = _compute_statistics_of_path_or_samples(path_or_samples2, model, batch_size, dims, device)
#     fid_value = calculate_frechet_distance(m1, s1, m2, s2)
#     return fid_value


