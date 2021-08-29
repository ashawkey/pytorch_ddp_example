#!pip install efficientnet_pytorch
import efficientnet_pytorch

import torch
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = efficientnet_pytorch.EfficientNet.from_pretrained("efficientnet-b7")
        n_features = self.net._fc.in_features
        self.net._fc = nn.Identity()

        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
        self.head = nn.Linear(in_features=n_features, out_features=1)
    
    def forward(self, x):
        # [B, 3, H, W]

        x = self.net(x)

        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.head(dropout(x))
            else:
                out += self.head(dropout(x))
        out /= len(self.dropouts)

        return out.squeeze(1)

