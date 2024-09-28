from lpips import LPIPS
import torch
import einops


@torch.no_grad()
class cal_LPIPS:
    '''
    gt and pred has the shape of 
        [n t c h w] (video)
        [n c h w] (image)
    '''
    def __init__(self, net='alex', use_gpu=False):
        assert net in ['alex', 'squeeze', 'vgg']
        self.lpips_model = LPIPS(net=net).eval()
        self.use_gpu = use_gpu
        if torch.cuda.is_available() and self.use_gpu:
            self.lpips_model = self.lpips_model.cuda()
        
    def __call__(self, pred, gt):
        pred = torch.maximum(pred, torch.min(gt))
        pred = torch.minimum(pred, torch.max(gt))
        if torch.cuda.is_available() and self.use_gpu:
            pred, gt = pred.cuda(), gt.cuda()
        else:
            pred, gt = pred.cpu(), gt.cpu()
        if len(pred.shape) == 5:
            pred = einops.rearrange(pred, 'n t c h w -> (n t) c h w')
            gt = einops.rearrange(gt, 'n t c h w -> (n t) c h w')
        return self.lpips_model(pred, gt).mean().detach().cpu()
