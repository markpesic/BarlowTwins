import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as tr
import torchvision.datasets as datasets
import numpy as np


from BarlowModel.barlowTwins import BarlowTwins
from BarlowModel.utils import criterion, get_byol_transforms
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

epochs = 30
batch_size = 256
offset_bs = 256
base_lr = 0.03

transformT, transformT1, transformEvalT = get_byol_transforms(32, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

#traindt = datasets.ImageNet(root='./data', split = 'train')
#trainloader = torch.utils.data.DataLoader(traindt, batch_size=batch_size, shuffle=True)

#testdt = datasets.ImageNet(root='./data', split = 'val')
#testloader = torch.utils.data.DataLoader(traindt, batch_size=128, shuffle=True)

trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=tr.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=tr.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

lr = base_lr*batch_size/offset_bs

model = BarlowTwins(input_size=512, output_size = 2048, backend='resnet18')

model.to(device)

params = model.parameters()
optimizer = optim.SGD( params, lr=lr, weight_decay=1.5e-6)
#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=)

def train_loop(model, optimizer, trainloader, transform, transform1, criterion, device):
    tk0 = tqdm(trainloader)
    train_loss = []

    for batch, _ in tk0:
        batch = batch.to(device)
        
        x = transform(batch)
        x1 = transform1(batch)

        fx = model(x)
        fx1 = model(x1)
        loss = criterion(fx, fx1)
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del batch, x, x1, fx, fx1
    return train_loss


for epoch in range(epochs):
    train_loss = train_loop(model, optimizer, trainloader, transformT, transformT1, criterion, torch.device('cuda'))
    print(np.mean(train_loss))







