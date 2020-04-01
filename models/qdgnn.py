import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from qpu_ops import *
from qpu_layers import *
from models.dgnn import GraphTemporalConv as RGraphTemporalConv 
from models.dgnn import BiTemporalConv as RBiTemporalConv
from models.dgnn import DGNBlock as RDGNBlock
class ARCCOS(nn.Module):
    def __init__(self,feat,dim):
        super(ARCCOS,self).__init__()
        self.feat = feat
        self.dim = dim
    def forward(self,x):
        x,i,j,k  = x.split(self.feat,self.dim)
        return torch.acos(torch.clamp(x,min=-1+1e-6,max=1-1e-6))

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        pad = int((kernel_size - 1) / 2)        
        self.conv = QuaternionProdConv1d(in_channels, out_channels, kernel_size, stride, pad)

    def forward(self, x):
        """
        x: (N, C, T, V)
        """
        N, C, T, V = x.shape
        x = x.permute(0, 3, 1, 2).reshape(N * V, C, T)
        x = self.conv(x)
        T = x.shape[-1]
        x = x.reshape(N, V, -1, T).permute(0, 2, 3, 1)
        return x


class BiTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        # NOTE: assuming that temporal convs are shared between node/edge features
        self.tempconv_e = TemporalConv(in_channels, out_channels, kernel_size, stride)
        self.tempconv_v = TemporalConv(in_channels, out_channels, kernel_size, stride)

    def forward(self, fv, fe):
        return self.tempconv_v(fv), self.tempconv_e(fe)


class DGNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, source_M, target_M):
        super().__init__()
        self.num_nodes, self.num_edges = source_M.shape
        # Adaptive block with learnable graphs; shapes (V_node, V_edge)
        # Not learnable for now...
        self.source_M = nn.Parameter(torch.from_numpy(source_M.astype('float32')))
        self.target_M = nn.Parameter(torch.from_numpy(target_M.astype('float32')))

        # Updating functions
        self.H_v = QPU(3 * in_channels, out_channels)
        self.H_e = QPU(3 * in_channels, out_channels)

    @staticmethod
    def quaternion_power_adj(f, A):
        r"""
        \prod_k f_ik ^ A_kj
        f: N, C, T, V1
        A: V1, V2
        out: N, C, T, V2
        """
        N, C, T, V1 = f.shape
        in_channels = C // 4
        _, V2 = A.shape
        f = f.unsqueeze(-1)  # N, C, T, V1, 1
        r, i, j, k = f.split(in_channels, 1)
        # Quaternion power
        r, i, j, k = quaternion_power(r, i, j, k, A+1e-6)
        # Chained prod
        r, i, j, k = QuaternionChainedProdFunction.apply(r, i, j, k, -2)
        f_out = torch.cat([r, i, j, k], dim=1)
        return quaternion_normalize(f_out, dim=1)

    def forward(self, fv, fe):
        # `fv` (node features) has shape (N, C, T, V_node)
        # `fe` (edge features) has shape (N, C, T, V_edge)
        N, C, T, V_node = fv.shape
        _, _, _, V_edge = fe.shape
        # Compute features for node/edge updates
        # fvp
        fe_in_agg = self.quaternion_power_adj(fe, self.source_M.transpose(0, 1))
        fe_out_agg = self.quaternion_power_adj(fe, self.target_M.transpose(0, 1))
        fvp = torch.stack((fv, fe_in_agg, fe_out_agg), dim=1)   # Out shape: (N, 3, C, T, V_node)
        fvp = fvp.reshape(N, 3*(C//4), 4, T, V_node).transpose(1, 2).reshape(N, 3*C, T, V_node).permute(0, 2, 3, 1) # (N, T, V_node, 3C)
        fvp = self.H_v(fvp).permute(0,3,1,2)    # (N,C_out,T,V_node)
        
        # fep
        fv_in_agg = self.quaternion_power_adj(fv, self.source_M)
        fv_out_agg = self.quaternion_power_adj(fv, self.target_M)
        fep = torch.stack((fe, fv_in_agg, fv_out_agg), dim=1)   # Out shape: (N, 3, C, T, V_edge)
        fep = fep.reshape(N, 3*(C//4), 4, T, V_edge).transpose(1, 2).reshape(N, 3*C, T, V_edge).permute(0, 2, 3, 1) # (N, T, V_edge, 3C)
        fep = self.H_e(fep).permute(0,3,1,2)    # (N,C_out,T,V_node)
        
        return fvp, fep


class GraphTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, source_M, target_M, temp_kernel_size=9, stride=1, residual=True, rinv=False):
        super(GraphTemporalConv, self).__init__()
        self.rinv = rinv
        if rinv:
            self.dgn = DGNBlock(in_channels, out_channels*4, source_M, target_M)
            self.scalar = AngleAxisMap(dim=1,rinv=rinv)
        else:
            self.dgn = DGNBlock(in_channels, out_channels, source_M, target_M)
            self.scalar = AngleAxisMap(,dim=1,rinv=rinv)
        self.tcn = RBiTemporalConv(out_channels, out_channels, kernel_size=temp_kernel_size, stride=stride)

    def forward(self, fv, fe):
        fv, fe = self.dgn(fv, fe)
        fv = self.scalar(fv)
        fe = self.scalar(fe)
        fv, fe = self.tcn(fv, fe)
        return fv, fe


class Model(nn.Module):
    def __init__(self, in_channels, num_joints, num_frames, num_cls, config):
        
        super(Model, self).__init__()
        num_class=num_cls
        num_point=num_joints
        num_person=2
        from models.directed_ntu_rgb_d import Graph
        self.graph = Graph(config['dataset'])

        source_M, target_M = self.graph.source_M, self.graph.target_M

       # Original
        self.l1 = GraphTemporalConv(in_channels, 64, source_M, target_M, residual=False, rinv=config['rinv'])
        self.l2 = RGraphTemporalConv(64, 64, source_M, target_M)
        self.l3 = RGraphTemporalConv(64, 64, source_M, target_M)
        self.l4 = RGraphTemporalConv(64, 64, source_M, target_M)
        self.l5 = RGraphTemporalConv(64, 128, source_M, target_M, stride=2)
        self.l6 = RGraphTemporalConv(128, 128, source_M, target_M)
        self.l7 = RGraphTemporalConv(128, 128, source_M, target_M)
        self.l8 = RGraphTemporalConv(128, 256, source_M, target_M, stride=2)
        self.l9 = RGraphTemporalConv(256, 256, source_M, target_M)
        self.l10 = RGraphTemporalConv(256, 256, source_M, target_M)

        self.fc = nn.Linear(256 * 2, num_class)

        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
    
    def forward(self, x):
        fv, fe = x

        N, C, T, V_node, M = fv.shape
        _, _, _, V_edge, _ = fe.shape

        # Preprocessing
        fv = fv.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V_node * C, T)
        fv = fv.view(N, M, V_node, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V_node)

        fe = fe.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V_edge * C, T)
        fe = fe.view(N, M, V_edge, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V_edge)

        fv, fe = self.l1(fv, fe)
        fv, fe = self.l2(fv, fe)
        fv, fe = self.l3(fv, fe)
        fv, fe = self.l4(fv, fe)
        fv, fe = self.l5(fv, fe)
        fv, fe = self.l6(fv, fe)
        fv, fe = self.l7(fv, fe)
        fv, fe = self.l8(fv, fe)
        fv, fe = self.l9(fv, fe)
        fv, fe = self.l10(fv, fe)

        # Shape: (N*M,C,T,V), C is same for fv/fe
        out_channels = fv.size(1)

        # Performs pooling over both nodes and frames, and over number of persons
        fv = fv.reshape(N, M, out_channels, -1).mean(3).mean(1)
        fe = fe.reshape(N, M, out_channels, -1).mean(3).mean(1)

        # Concat node and edge features
        out = torch.cat((fv, fe), dim=-1)

        return self.fc(out)

    def get_loss(self, input, target):
        return torch.nn.functional.cross_entropy(input, target)


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    model = Model(graph='graph.directed_ntu_rgb_d.Graph')

    print('Model total # params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
