# modified from https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py#L93

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torchvision.transforms.functional import center_crop

from torchvision.models.inception import inception_v3, Inception_V3_Weights
import einops
import numpy as np
from scipy.stats import entropy

@torch.no_grad()
class ISProbaExtractor:
    def __init__(self, use_gpu=False, resize_crop=False):
        self.use_gpu = use_gpu
        self.resize_crop = resize_crop
        self.inception = inception_v3(
            weights=Inception_V3_Weights.DEFAULT, transform_input=False
        ).eval()
        if torch.cuda.is_available() and self.use_gpu:
            self.inception = self.inception.cuda()
            
    def __call__(self, images_fake, images_real):
        images_fake = self.bilinear_interpolation(images=images_fake)
        if torch.cuda.is_available() and self.use_gpu:
            images_fake = images_fake.cuda()
        else:
            images_fake = images_fake.cpu()
        if images_real.shape[-3] == 1:  # n c h w
            images_fake = images_fake.repeat(1, 3, 1, 1)
        logits_fake = F.softmax(self.inception(images_fake), dim=-1).detach().cpu()
        # print(logits_fake.shape)
        return logits_fake
        
    
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


def compute_IS(proba, splits=1):
    # Now compute the mean kl-div
    N = proba.shape[0]
    scores = []
    for k in range(splits):
        part = proba[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
    return np.mean(np.exp(np.mean(scores))), np.std(np.exp(np.mean(scores)))



def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)
