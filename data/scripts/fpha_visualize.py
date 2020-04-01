import os
import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np

from data.fpha import fpha_info
from data.fpha.feeder import Feeder
from utils.skel_utils.skeleton import Skeleton
from utils.skel_utils.visualize import skeleton_anim

NUM_JOINTS = 21

def fpha_skeleton_anim():
    """Create an animation of skeleton in ntu dataset
    """
    root = '/media/xuan/ssd/data/fpha_processed'
    xyz_data_path = os.path.join(root, 'xyz', 'train_data.npy')
    xyz_label_path = os.path.join(root, 'xyz','train_label.pkl')
    xyz_data = Feeder(xyz_data_path, xyz_label_path, num_samples=-1, num_frames=20, mmap=True)
    colors = [''] * (NUM_JOINTS - 1)
    for i in range(NUM_JOINTS - 1):
        if i + 1 in fpha_info.T:
            colors[i] = 'r'
        elif i + 1 in fpha_info.I:
            colors[i] = 'g'
        elif i + 1 in fpha_info.M:
            colors[i] = 'b'
        elif i + 1 in fpha_info.R:
            colors[i] = 'c'
        elif i + 1 in fpha_info.P:
            colors[i] = 'm'

    np.random.seed()
    for _ in range(20):
        # Prepare data
        xyz, label = xyz_data[np.random.randint(len(xyz_data))]
        # Prepare fig and axe
        fig = plt.figure(fpha_info.LABEL_NAMES[label])
        ax = fig.add_subplot(1, 1, 1, projection='3d', xlabel='X', ylabel='Y', zlabel='Z')
        ax.view_init(azim=-30, elev=-60)
        # Add animation    
        anim = skeleton_anim(fig, ax, xyz, fpha_info.PARENTS, fps=10, colors=colors)
        anim.save('data/plots/fpha/xyz.gif', writer='imagemagick', fps=10)
        # Show
        plt.show()


def fpha_compare():
    """Create side by side animation of xyz and quaternion reconstruction
    """
    root = '/media/xuan/ssd/data/fpha_processed'
    num_frames = 20
    xyz_data_path = os.path.join(root,'xyz', 'train_data.npy')
    xyz_label_path = os.path.join(root, 'xyz', 'train_label.pkl')
    xyz_data = Feeder(xyz_data_path, xyz_label_path, num_samples=-1, num_frames=num_frames, mmap=True)
    
    q_data_path = os.path.join(root, 'quaternion', 'train_data.npy')
    q_label_path = os.path.join(root, 'quaternion', 'train_label.pkl')
    q_data = Feeder(q_data_path, q_label_path, num_samples=-1, num_frames=num_frames, mmap=True)
    
    qrec_data_path = os.path.join(root, 'qrec', 'train_data.npy')
    qrec_label_path = os.path.join(root, 'qrec', 'train_label.pkl')
    qrec_data = Feeder(qrec_data_path, qrec_label_path, num_samples=-1, num_frames=num_frames, mmap=True)
    
    colors = [''] * (NUM_JOINTS - 1)
    for i in range(NUM_JOINTS - 1):
        if i + 1 in fpha_info.T:
            colors[i] = 'r'
        elif i + 1 in fpha_info.I:
            colors[i] = 'g'
        elif i + 1 in fpha_info.M:
            colors[i] = 'b'
        elif i + 1 in fpha_info.R:
            colors[i] = 'c'
        elif i + 1 in fpha_info.P:
            colors[i] = 'm'

    np.random.seed()
    sk = Skeleton(parents=fpha_info.PARENTS)
    for _ in range(20):
        # Prepare data
        i = np.random.randint(len(xyz_data))
        # xyz
        xyz, label = xyz_data[i]
        # Loaded qrec
        qrec, _ = qrec_data[i]  # Loaded qrec
        # Computed from loaded quaternion
        q, _ = q_data[i]
        qrec_computed = sk.qu2xyz(q.transpose(1, 0, 2), fpha_info.AVG_BONE_LENS)
        qrec_computed = qrec_computed.transpose(1, 0, 2)
        # Prepare fig and axes
        title = fpha_info.LABEL_NAMES[label] + ' (xyz)'
        fig1 = plt.figure(title, figsize=(5, 5))
        ax1 = fig1.add_subplot(1, 1, 1, projection='3d', xlabel='X', ylabel='Y', zlabel='Z')
        ax1.set_title(title, loc='left')
        ax1.view_init(azim=-30, elev=-60)    
        
        title = fpha_info.LABEL_NAMES[label] + ' (qrec loaded)'
        fig2 = plt.figure(title, figsize=(5, 5))
        ax2 = fig2.add_subplot(1, 1, 1, projection='3d', xlabel='X', ylabel='Y', zlabel='Z')
        ax2.set_title(title, loc='left')
        ax2.view_init(azim=-30, elev=-60)    
        
        title = fpha_info.LABEL_NAMES[label] + ' (qrec from q)'
        fig3 = plt.figure(title, figsize=(5, 5))
        ax3 = fig3.add_subplot(1, 1, 1, projection='3d', xlabel='X', ylabel='Y', zlabel='Z')
        ax3.set_title(title, loc='left')
        ax3.view_init(azim=-30, elev=-60)    
        
        # Add animiation
        anim1 = skeleton_anim(fig1, ax1, xyz, fpha_info.PARENTS, fps=10, colors=colors)
        anim2 = skeleton_anim(fig2, ax2, qrec, fpha_info.PARENTS, fps=10, colors=colors)
        anim3 = skeleton_anim(fig3, ax3, qrec_computed, fpha_info.PARENTS, fps=10, colors=colors)
        anim1.save('data/plots/fpha/fpha_xyz.gif', writer='imagemagick', fps=10)
        anim2.save('data/plots/fpha/fpha_qrec_loaded.gif', writer='imagemagick', fps=10)
        anim3.save('data/plots/fpha/fpha_qrec_from_q.gif', writer='imagemagick', fps=10)

        # Show
        plt.show()


if __name__ == '__main__':
    function = sys.argv[1]
    if function == 'single':
        fpha_skeleton_anim()
    elif function == 'compare':
        fpha_compare()
    else:
        ValueError()