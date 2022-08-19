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
    tr.RandomResizedCrop(size=size, scale=(0.08,1), ratio=(3 / 4, 4 / 3)),
    tr.RandomApply(nn.ModuleList([tr.RandomRotation((-90, 90))]), p=0.5),
    tr.RandomApply(nn.ModuleList([tr.ColorJitter()]), p=0.8),
    tr.GaussianBlur(kernel_size=(23,23), sigma=(0.1, 2.0)),
    #tr.RandomGrayscale(p=0.2),
    tr.Normalize(mean, std)])

    transformT1 = tr.Compose([
        tr.RandomResizedCrop(size=size, scale=(0.08,1), ratio=(3 / 4, 4 / 3)),
        tr.RandomApply(nn.ModuleList([tr.RandomRotation((-90, 90))]), p=0.5),
        tr.RandomApply(nn.ModuleList([tr.ColorJitter()]), p=0.8),
        #tr.RandomGrayscale(p=0.2),
        tr.RandomApply(nn.ModuleList([tr.GaussianBlur(kernel_size=(23,23), sigma=(0.1, 2.0))]), p=0.1),
        tr.Normalize(mean, std)])

    transformEvalT = tr.Compose([
        tr.CenterCrop(size=size),
        tr.Normalize(mean, std)
    ])

    return transformT, transformT1, transformEvalT