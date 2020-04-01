import torch
import torch.nn as nn

from qpu_layers import *


class RLSTM(nn.Module):
    def __init__(self, in_channels, num_joints, num_frames, num_cls, config):
        super(RLSTM, self).__init__()
        self.dev = config['device_ids'][0]
        self.num_joints = num_joints
        self.num_frames = num_frames
        self.feat_dim = 256

        self.mlp = nn.Sequential(
            nn.Linear(self.num_joints * 4, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(input_size=self.feat_dim, hidden_size=self.feat_dim, num_layers=1)

        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.feat_dim, num_cls)
        )

    def forward(self, x):
        """
        Args:
            x: torch.Tensor(B, 4, F, J, M)
        """
        batch_size = x.shape[0]
        num_person=x.shape[-1]
        x = x.permute(0, 4, 2, 1, 3)  # B, M, F, 4, J
        x = x.reshape(batch_size * num_person * self.num_frames, self.num_joints * 4)
        x = self.mlp(x)  # B*M*F, C*4
        x = x.reshape(batch_size*num_person, self.num_frames, -1)  # (B*M, F, C*4)
    
        x = x.permute(1, 0, 2)  # (F, B*M, C*4)
        h0 = torch.zeros(1, batch_size*num_person, self.feat_dim, device=self.dev)
        c0 = torch.zeros(1, batch_size*num_person, self.feat_dim, device=self.dev)
        output, (hn, cn) = self.lstm(x, (h0, c0))  # out: (F, B*M, C)
        output = output.permute(1, 0, 2).reshape(batch_size, num_person * self.num_frames, -1)
        x = torch.mean(output, dim=1)
        x = self.classifier(x)
        return x

    def get_loss(self, input, target):
        return torch.nn.functional.cross_entropy(input, target)


class QLSTM(nn.Module):
    def __init__(self, in_channels, num_joints, num_frames, num_cls, config):
        super(QLSTM, self).__init__()
        self.config = config
        self.num_joints = num_joints
        self.num_frames = num_frames
        self.feat_dim = 256
        if 'rinv' in config and config['rinv']:
            self.mlp = nn.Sequential(
                QPU(self.num_joints * 4, self.feat_dim),
                QPU(self.feat_dim, self.feat_dim*4),
                QuaternionScalarMap(self.feat_dim,dim=-1,rinv=True)
            )
        else:
            self.mlp = nn.Sequential(
                QPU(self.num_joints * 4, self.feat_dim),
                QPU(self.feat_dim, self.feat_dim),
                QuaternionScalarMap(self.feat_dim//4,dim=-1,rinv=False)
            )

        self.lstm = nn.LSTM(input_size=self.feat_dim, hidden_size=self.feat_dim, num_layers=1)

        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.feat_dim, num_cls)
        )

    def forward(self, x):
        """
        Args:
            x: torch.Tensor(B, 4, F, J, M)
        """
        batch_size = x.shape[0]
        num_person=x.shape[-1]
        x = x.permute(0, 4, 2, 1, 3)  # B, M, F, 4, J
        x = x.reshape(batch_size * num_person * self.num_frames, self.num_joints * 4)
        x = self.mlp(x)  # B*M*F, C*4
        x = x.reshape(batch_size*num_person, self.num_frames, -1)  # (B*M, F, C*4)
    
        x = x.permute(1, 0, 2)  # (F, B*M, C*4)
        h0 = torch.zeros(1, batch_size*num_person, self.feat_dim, device=self.config['device_ids'][0])
        c0 = torch.zeros(1, batch_size*num_person, self.feat_dim, device=self.config['device_ids'][0])
        output, (hn, cn) = self.lstm(x, (h0, c0))  # out: (F, B*M, C)
        output = output.permute(1, 0, 2).reshape(batch_size, num_person * self.num_frames, -1)
        x = torch.mean(output, dim=1)
        x = self.classifier(x)
        return x

    def get_loss(self, input, target):
        return torch.nn.functional.cross_entropy(input, target)
        