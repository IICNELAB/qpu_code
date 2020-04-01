import sys
import os
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np

from data.ntu import ntu_info
from data.ntu.feeder import Feeder
from utils.skel_utils.skeleton import Skeleton
from utils.skel_utils.visualize import skeleton_anim

NUM_JOINTS = 25
COLORS = ['b'] * NUM_JOINTS
for i in range(NUM_JOINTS):
    if i in ntu_info.LEFT_HAND:
        COLORS[i] = 'r'
    elif i in ntu_info.RIGHT_HAND:
        COLORS[i] = 'g'
    elif i in ntu_info.SPINE:
        COLORS[i] = 'b'
    elif i in ntu_info.LEFT_LEG:
        COLORS[i] = 'c'
    elif i in ntu_info.RIGHT_LEG:
        COLORS[i] = 'm'


def ntu_anim_xyz(seed=None):
    """Create an animation of skeleton in ntu dataset
    """
    root = '../../Dataset/NTU-RGB-D'
    benchmark = 'xview_pre'
    xyz_data_path = os.path.join(root, 'xyz', benchmark, 'train_data.npy')
    xyz_label_path = os.path.join(root, 'xyz', benchmark, 'train_label.pkl')
    xyz_data = Feeder(xyz_data_path, xyz_label_path, num_samples=-1, num_frames=20, mmap=True)
    # Prepare data
    np.random.seed(seed)
    index = np.random.randint(len(xyz_data))
    xyz, label = xyz_data[index]
    xyz = xyz[..., 0]
    # Prepare fig and axe
    fig = plt.figure(ntu_info.LABEL_NAMES[label])
    ax = fig.add_subplot(1, 1, 1, projection='3d', xlabel='X', ylabel='Y', zlabel='Z')
    ax.view_init(azim=90, elev=-70)
    # Add animation    
    anim = skeleton_anim(fig, ax, xyz, ntu_info.PARENTS, fps=10, colors=COLORS)
    anim.save('data/plots/ntu/xyz.gif', writer='imagemagick', fps=10)
    # Show()
    plt.show()


def ntu_anim_qabs(seed=None):
    """Create an animation of skeleton in ntu dataset
    """
    root = '/media/xuan/ssd/data/NTU-RGB-D-pre'
    benchmark = 'xsub'
    modality='quaternion'
    data_path = os.path.join(root, modality, benchmark, 'train_data.npy')
    label_path = os.path.join(root, modality, benchmark, 'train_label.pkl')
    data = Feeder(data_path, label_path, num_samples=-1, num_frames=20, mmap=True)
    sk = Skeleton(parents=ntu_info.PARENTS)
    # Prepare data
    np.random.seed(seed)
    index = np.random.randint(len(data))
    quaternion, label = data[index]
    quaternion = quaternion[..., 0]
    xyz = sk.qabs2xyz(quaternion.transpose(1, 0, 2), ntu_info.AVG_BONE_LENS).transpose(1, 0, 2)
    # Prepare fig and axe
    fig = plt.figure(ntu_info.LABEL_NAMES[label])
    ax = fig.add_subplot(1, 1, 1, projection='3d', xlabel='X', ylabel='Y', zlabel='Z')
    ax.view_init(azim=90, elev=-70)
    # Add animation    
    anim = skeleton_anim(fig, ax, xyz, ntu_info.PARENTS, fps=10, colors=COLORS)
    anim.save('data/plots/ntu/qabs.gif', writer='imagemagick', fps=10)
    # Show()
    plt.show()

def ntu_anim_qrel(seed=None):
    """Create an animation of skeleton in ntu dataset
    """
    root = '/media/xuan/ssd/data/NTU-RGB-D-pre'
    benchmark = 'xsub'
    modality='qrel'
    data_path = os.path.join(root, modality, benchmark, 'val_data.npy')
    label_path = os.path.join(root, modality, benchmark, 'val_label.pkl')
    data = Feeder(data_path, label_path, num_samples=-1, num_frames=20, mmap=True)
    sk = Skeleton(parents=ntu_info.PARENTS)
    # Prepare data
    np.random.seed(seed)
    index = np.random.randint(len(data))
    quaternion, label = data[index]
    quaternion = quaternion[..., 0]
    xyz = sk.qrel2xyz(quaternion.transpose(1, 0, 2), ntu_info.AVG_BONE_LENS).transpose(1, 0, 2)
    # Prepare fig and axe   
    fig = plt.figure(ntu_info.LABEL_NAMES[label])
    ax = fig.add_subplot(1, 1, 1, projection='3d', xlabel='X', ylabel='Y', zlabel='Z')
    ax.view_init(azim=90, elev=-70)
    # Add animation    
    anim = skeleton_anim(fig, ax, xyz, ntu_info.PARENTS, fps=10, colors=COLORS)
    anim.save('data/plots/ntu/qrel.gif', writer='imagemagick', fps=10)
    # Show()
    plt.show()


if __name__ == '__main__':
    function = sys.argv[1]
    seed = 1011
    if function == 'xyz':
        ntu_anim_xyz(seed)
    elif function == 'qabs':
        ntu_anim_qabs(seed)
    elif function == 'qrel':
        ntu_anim_qrel(seed)
    elif function == 'all':
        np.random.seed()
        seed=np.random.randint(100)
        ntu_anim_xyz(seed)
        # ntu_anim_qabs(seed)
        ntu_anim_qrel(seed)
    else:
        raise ValueError()