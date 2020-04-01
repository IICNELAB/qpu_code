from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
from torch.autograd import Function
import torch.nn as nn
import etw_pytorch_utils as pt_utils
import sys
import math

try:
    import builtins
except:
    import __builtin__ as builtins

try:
    import pointnet2._ext as _ext
except ImportError:
    if not getattr(builtins, "__POINTNET2_SETUP__", False):
        raise ImportError(
            "Could not import _ext module.\n"
            "Please see the setup instructions in the README: "
            "https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/README.rst"
        )

if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *


class RandomDropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(RandomDropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, X):
        theta = torch.Tensor(1).uniform_(0, self.p)[0]
        return pt_utils.feature_dropout_no_scaling(X, theta, self.train, self.inplace)


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        return _ext.furthest_point_sampling(xyz, npoint)

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor

        idx : torch.Tensor
            (B, npoint) tensor of the features to gather

        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """

        _, C, N = features.size()

        ctx.for_backwards = (idx, C, N)

        return _ext.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards

        grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


gather_operation = GatherOperation.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        # type: (Any, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
            Find the three nearest neighbors of unknown in known
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known features
        known : torch.Tensor
            (B, m, 3) tensor of unknown features

        Returns
        -------
        dist : torch.Tensor
            (B, n, 3) l2 distance to the three nearest neighbors
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbors
        """
        dist2, idx = _ext.three_nn(unknown, known)

        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        r"""
            Performs weight linear interpolation on 3 features
        Parameters
        ----------
        features : torch.Tensor
            (B, c, m) Features descriptors to be interpolated from
        idx : torch.Tensor
            (B, n, 3) three nearest neighbors of the target features in features
        weight : torch.Tensor
            (B, n, 3) weights

        Returns
        -------
        torch.Tensor
            (B, c, n) tensor of the interpolated features
        """
        B, c, m = features.size()
        n = idx.size(1)

        ctx.three_interpolate_for_backward = (idx, weight, m)

        return _ext.three_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, c, n) tensor with gradients of ouputs

        Returns
        -------
        grad_features : torch.Tensor
            (B, c, m) tensor with gradients of features

        None

        None
        """
        idx, weight, m = ctx.three_interpolate_for_backward

        grad_features = _ext.three_interpolate_grad(
            grad_out.contiguous(), idx, weight, m
        )

        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of features to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()

        ctx.for_backwards = (idx, N)

        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, N = ctx.for_backwards

        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, None


grouping_operation = GroupingOperation.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        return _ext.ball_query(new_xyz, xyz, radius, nsample)

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """

        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


class GroupAll(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz=True):
        # type: (GroupAll, bool) -> None
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (GroupAll, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features
        
### Code for quaternion representation

def normalize(v, dim):
    n = torch.sqrt(torch.sum(v*v, dim=dim, keepdim=True))
    return v / (n + 1e-6)


def project(p, v, dim):
    """
    project v to the plan orthognal to p
    p is normalized
    """
    p = normalize(p, dim)
    vert = torch.sum(p * v, dim=dim, keepdim=True) * p
    return v - vert
    

def project_one(p, dim):
    """Get one reference vector in the orthogonal projection plan.
       [1,0,0] - <[1,0,0],p>*p = [1-px*px, -px*py, -px*pz] or
       [0,1,0] - <[0,1,0],p>*p = [-py*px, 1-py*py, -py*pz] if p colinear with [1,0,0]
       Then normalize
    """
    p = normalize(p, dim)
    ref = torch.zeros_like(p)
    px = p.select(dim=dim, index=0)
    py = p.select(dim=dim, index=1)
    pz = p.select(dim=dim, index=2)
    colinear_x = torch.abs(px) > (1 - 1e-3)
    ref.select(dim, 0)[~colinear_x] = 1 - px[~colinear_x] * px[~colinear_x]
    ref.select(dim, 1)[~colinear_x] = - px[~colinear_x] * py[~colinear_x]
    ref.select(dim, 2)[~colinear_x] = - px[~colinear_x] * pz[~colinear_x]
    # if colinear
    ref.select(dim, 0)[colinear_x] = - py[colinear_x] * px[colinear_x]
    ref.select(dim, 1)[colinear_x] = 1 - py[colinear_x] * py[colinear_x]
    ref.select(dim, 2)[colinear_x] = - py[colinear_x] * pz[colinear_x]
    return normalize(ref, dim)
    

def rot_sort(p, pts, coord_dim, sample_dim, ref=None):
    """
    sort pts according to their orthogonal projection of p, 
    clockwise w.r.t one reference vector.
    """
    p = normalize(p, dim=coord_dim)
    
    if ref is None:
        ref = project_one(p, coord_dim)
    ref = ref.expand_as(pts)
    
    projs = normalize(project(p, pts, coord_dim), coord_dim)

    # Compute angles from ref to projs 
    sinus = torch.sum(torch.cross(ref, projs, coord_dim) * p, dim=coord_dim, keepdim=True)
    cosinus = torch.sum(ref * projs, dim=coord_dim, keepdim=True)
    angles = torch.atan2(sinus, cosinus)

    # If projection is too small, we randomly give an angle 
    # (because ref is not rotation-invariant)
    close_ind = torch.sum(projs * projs, dim=coord_dim, keepdim=True) < 1e-12
    angles[close_ind] = (torch.rand(close_ind.sum()).cuda() - 0.5) * math.pi * 2

    # Sort according to angles
    ind = angles.argsort(dim=sample_dim)
    pts = pts.gather(index=ind.expand_as(pts), dim=sample_dim)
    
    return pts


def dist_sort(pts, coord_dim, sample_dim):
    """
    Sort according to distance to origin
    """
    dist = torch.sqrt(torch.sum(pts * pts, dim=coord_dim, keepdim=True))

    # Sort according to dist
    ind = dist.argsort(dim=sample_dim)
    pts = pts.gather(index=ind.expand_as(pts), dim=sample_dim)

    return pts


def to_quat(xyz, radius):
    """
    xyz: B, 3, N, S -> B, 4, N, S
    """
    dist = torch.sqrt(torch.sum(xyz * xyz, dim=1, keepdim=True))

    ori = xyz / (dist + 1e-6)
    theta = dist / radius * math.pi / 2.
    s = torch.cos(theta)
    v = torch.sin(theta) * ori
    q = torch.cat([s, v], dim=1)

    return q


class QueryAndGroupQuat(nn.Module):
    r"""
    Groups with a ball query of radius, then convert neighbor points to quaternion, sorted by distance

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_center):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroupQuat, self).__init__()
        self.radius, self.nsample, self.use_center = radius, nsample, use_center

    def forward(self, xyz, new_xyz, features=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """

        # idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        idx = ball_query(self.radius, self.nsample + 1, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz = grouped_xyz[:, :, :, 1:]

        new_xyz = new_xyz.transpose(1, 2).unsqueeze(-1)  # B, 3, npoint, 1
        grouped_xyz -= new_xyz
        
        grouped_xyz = rot_sort(p=new_xyz, pts=grouped_xyz, coord_dim=1, sample_dim=-1)
        # grouped_xyz = dist_sort(pts=grouped_xyz, coord_dim=1, sample_dim=-1)

        B, _, npoint, nsample = grouped_xyz.shape

        grouped_quat = to_quat(grouped_xyz, self.radius).unsqueeze(2)  # B, 4, 1, npoint, nsample
        new_features = [grouped_quat]

        # Cyclic permute and concat
        M = 8
        index = torch.tensor([x for x in range(1, nsample)] + [0]).cuda()
        for _ in range(M - 1):
            new_features.append(new_features[-1].index_select(dim=-1, index=index))

        new_features = torch.cat(new_features, dim=2)  # B, 4, M, npoint, nsample
        new_features = new_features.reshape(B, 4 * M, npoint, nsample)

        return new_features