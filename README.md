# BarlowTwins
## A Barlow Twins Implementation in pytorch [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/pdf/2103.03230.pdf)

![Barlow Twins architecure](https://github.com/markpesic/BarlowTwins/blob/master/images/barlowTwins.png?raw=true)

## Model

```python
from BarlowModel.barlowTwins import BarlowTwins
model = BarlowTwins(
    input_size=2048,
    output_size=8192,
    depth_projector=3,
    backend='resnet50',
    pretrained=False)
   
```

## Training
```python
import torch
from BarlowModel.barlowTwins import BarlowTwins
from BarlowModel.utils import criterion, get_byol_transforms
#train_loader, size, mean, std, lr and device given by the users
t, t1, _ = get_byol_transforms(size, mean, std)
model = BarlowTwins(
    input_size=2048,
    output_size=8192,
    depth_projector=3,
    backend='resnet50',
    pretrained=False)
    
model = model.to(device)
optimizer = torch.optim.SGD( model.parameters(), lr=lr, momentum= 0.9, weight_decay=1.5e-4)
for epoch in range(30):
    model.train()
    for batch, _ in train_loader:
        batch = batch.to(device)
        x = t(batch)
        x1 = t1(batch)
        fx = model(x)
        fx1 = model(x1)
        loss = criterion(fx, fx1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Citation 
```bibtex
@misc{https://doi.org/10.48550/arxiv.2103.03230,
  doi = {10.48550/ARXIV.2103.03230},
  
  url = {https://arxiv.org/abs/2103.03230},
  
  author = {Zbontar, Jure and Jing, Li and Misra, Ishan and LeCun, Yann and Deny, Stéphane},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), Neurons and Cognition (q-bio.NC), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Biological sciences, FOS: Biological sciences},
  
  title = {Barlow Twins: Self-Supervised Learning via Redundancy Reduction},
  
  publisher = {arXiv},
  
  year = {2021},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
