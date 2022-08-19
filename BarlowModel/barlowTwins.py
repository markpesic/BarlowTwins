from turtle import back
import torch
import torch.nn as nn
import torch.nn.functional as F
from BarlowModel.Backend import resnet18, resnet34, resnet50, resnet101, resnet152

availableBackends = {'resnet18':resnet18, 'resnet34':resnet34, 'resnet50':resnet50, 'resnet101':resnet101,'resnet152':resnet152}

class MLP(nn.Module):
    def __init__(self,
    input_size = 2048,
    output_size = 8192,
    depth = 3,
    ):  
        super().__init__()
        layers = []
        inp = input_size
        for d in range(depth):
            if d == depth - 1:
                layers.append(nn.Linear(inp, output_size))
            else:
                layers.extend([nn.Linear(inp, output_size), nn.BatchNorm1d(output_size), nn.ReLU(inplace=True)])
                inp = output_size
        self.layer = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)

class BarlowTwins(nn.Module):
    def __init__(self,
    backend = 'resnet50',
    input_size = 2048,
    output_size = 8192,
    depth_projector = 3,
    pretrained_backend=False):
        super().__init__()

        self.backend = availableBackends[backend](pretrained=pretrained_backend)
        self.projector = MLP(input_size=input_size, output_size=output_size, depth=depth_projector)

    def forward(self, x):
        x = self.backend(x)
        x = self.projector(x)

        return x