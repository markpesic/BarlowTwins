import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tr

def criterion(x, y, lmbd = 5e-3):
    bs = x.size(0)
    emb = x.size(1)

    xNorm = (x - x.mean(0)) / x.std(0)
    yNorm = (y - y.mean(0)) / y.std(0)
    crossCorMat = (xNorm.T@yNorm) / bs
    loss = (crossCorMat*lmbd - torch.eye(emb, device=torch.device('cuda'))*lmbd).pow(2)
    
    return loss.sum()

def get_byol_transforms(size, mean, std):
    transformT = tr.Compose([
        transforms.ToTensor(),
        tr.RandomResizedCrop(size=size, scale=(0.08,1), ratio=(3 / 4, 4 / 3)),
        tr.RandomRotation((-90, 90)),
        tr.ColorJitter(),
        tr.GaussianBlur(kernel_size=(23,23), sigma=(0.1, 2.0)),
        tr.RandomGrayscale(p=0.2),
        tr.Normalize(mean, std),
        ])

    transformT1 = tr.Compose([
        transforms.ToTensor(),
        tr.RandomResizedCrop(size=size, scale=(0.08,1), ratio=(3 / 4, 4 / 3)),
        tr.RandomRotation((-90, 90)),
        tr.ColorJitter(),
        tr.RandomGrayscale(p=0.2),
        tr.GaussianBlur(kernel_size=(23,23), sigma=(0.1, 2.0)),
        tr.Normalize(mean, std),
        ])

    transformEvalT = tr.Compose([
        transforms.ToTensor(),
        tr.CenterCrop(size=size),
        tr.Normalize(mean, std),
        
    ])

    return transformT, transformT1, transformEvalT

from torchvision.transforms import transforms


class MultiViewDataInjector(object):
    def __init__(self, *args):
        self.transforms = args[0]
        self.random_flip = transforms.RandomHorizontalFlip()

    def __call__(self, sample, *with_consistent_flipping):
        if with_consistent_flipping:
            sample = self.random_flip(sample)
        output = [transform(sample) for transform in self.transforms]
        return output