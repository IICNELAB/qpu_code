import torch
import torch.nn as nn
from qpu_layers import *


class MLPBase(nn.Module):
    def __init__(self):
        super(MLPBase, self).__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size, -1)
        return self.mlp(x)


class RMLP(MLPBase):
    def __init__(self, num_data, num_cls, in_channel=4):
        super(RMLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(in_channel * num_data, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 128), 
                                 nn.ReLU(),
                                 nn.Linear(128, num_cls)
                                )


class QMLP(MLPBase):
    def __init__(self, num_data, num_cls, in_channel=4):
        super(QMLP, self).__init__()
        self.mlp = nn.Sequential(QPU(in_channel * num_data, 128),
                                 QPU(128, 128),
                                 nn.Linear(128, num_cls)
                                )


class QMLP_RInv(MLPBase):
    def __init__(self, num_data, num_cls, in_channel=4):
        super(QMLP_RInv, self).__init__()
        self.mlp = nn.Sequential(QPU(in_channel * num_data, 128),
                                 QPU(128, 128*4),
                                 KeepRealPart(dim=-1),
                                 nn.Linear(128, num_cls)
                                )

