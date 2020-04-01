import os
import sys
sys.path.append('.')

import pickle
import argparse
import numpy as np
from numpy.lib.format import open_memmap

from tqdm import tqdm
import random
from shutil import copyfile

from data.fpha import fpha_info
from utils.skel_utils.skeleton import Skeleton


# Dataset settings
NUM_JOINT = 21
MAX_FRAME = 300


def collect_samples(data_path, part='val'):
    """Collect selected sample names and labels.
        data_path: (str) directory containing skeleton files
    Args: 
        ignored_sample_path: (str) skeleton files to be ignored
        sub_samples: (int) number of samples to be subsampled. None if no subsampling.
        benchmark: (str) 'xview' or 'xsub'
        part: (str) 'val' or 'train' 
    return:
        train_sample_name: (list(str)) list of selected train sample names
        train_sample_label: (list(int) list of selected train sample labels
        test_sample_name: (list(str)) list of selected test sample names
        test_sample_label: (list(int) list of selected test sample labels
    """
    split_file = os.path.join(data_path, 'data_split_action_recognition.txt')
    # Get train and test file path and labels from `data_split_action_recognition.txt`
    sample_name = []
    sample_label = []
    with open(split_file, 'r') as f:
        # Train data
        line = f.readline().split(' ')
        _, num_samples = line[0], int(line[1])
        for _ in range(num_samples):
            line = f.readline().split(' ')
            if part == 'train':
                path, label = line[0], int(line[1])
                sample_name.append(path)
                sample_label.append(label)
        # Test data
        if part == 'val':
            line = f.readline().split(' ')
            _, num_samples = line[0], int(line[1])
            for _ in range(num_samples):
                line = f.readline().split(' ')
                path, label = line[0], int(line[1])
                sample_name.append(path)
                sample_label.append(label)
    return sample_name, sample_label


def read_skeleton(sample_name, skeleton_root):
    """
    Read a skeleton from file
    Args:
        sample_name: e.g. Subject_1/drink_mug/1
        skeleton_root: folder containing skeleton files
    Return:
        skeleton: np.array(3, num_frames, joints)
        label: int
    """
    skeleton_path = os.path.join(skeleton_root, sample_name, 'skeleton.txt')
    skeleton_vals = np.loadtxt(skeleton_path)
    skeleton = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21, -1)  # num_frames, joints, 3
    skeleton = np.transpose(skeleton, (2, 0, 1))  # 3, num_frames, joints
    num_frames = skeleton.shape[1]
    # Subsample if too long
    if num_frames > MAX_FRAME:
        print(sample_name, num_frames)
        interval = np.max([num_frames // MAX_FRAME, 1])
        ind = [interval * i for i in range(MAX_FRAME)]
        skeleton = skeleton.take(ind, axis=1)
        num_frames = MAX_FRAME
    return skeleton, num_frames


def gen_xyz(data_path, out_path, part, sample_name, sample_label):
    """Read skeleton data and save to file.
    Args: 
        data_path: (str) directory containing skeleton files
        out_path: (str) directory to save processed data
        part: (str) 'train' or 'eval'
        sample_name: (list(str)) list of selected sample names
        sample_label: (list(int)) list of selected sample labels
    """
    # Open memory to write
    data_save_path = os.path.join(out_path, '{}_data.npy'.format(part))
    fp = open_memmap(data_save_path, dtype='float32', mode='w+',
                    shape=(len(sample_label), 3, MAX_FRAME, NUM_JOINT))
    num_frame_list = []
    # Process and save joint data
    for i, s in tqdm(enumerate(sample_name), total=len(sample_name)):
        data, num_frame = read_skeleton(s, data_path)
        fp[i, :, :data.shape[1], :] = data
        num_frame_list.append(num_frame)
    # Save label and num_frame
    label_save_path = os.path.join(out_path, '{}_label.pkl'.format(part))
    with open(label_save_path, 'wb') as f:
        pickle.dump((sample_name, list(sample_label), list(num_frame_list)), f)

def xyz2qrel(xyz_path, out_path, part):
    """Convert xyz data to quaternion data (angle between bone and Z axis)
    Args:
        xyz_path: (str) directory of processed xyz data
        out_path: (str) directory to save converted quaternion data
        part: (str) 'train' or 'eval' 
    """
    q_data_path = os.path.join(out_path, '{}_data.npy'.format(part))
    q_label_path = os.path.join(out_path, '{}_label.pkl'.format(part))
    xyz_data_path = os.path.join(xyz_path, '{}_data.npy'.format(part))
    xyz_label_path = os.path.join(xyz_path, '{}_label.pkl'.format(part))
    if not os.path.exists(q_label_path):
        copyfile(xyz_label_path, q_label_path)
    # Read xyz data (num_samples, 3, num_frame, num_joint)
    xyz_data = np.load(xyz_data_path, mmap_mode='r')
    num_sample, _, max_frame, num_joint = xyz_data.shape
    max_body = 1
    # Get num_frame
    with open(q_label_path, 'rb') as f:
        _, _, num_frame = pickle.load(f)
    # Open memory to write
    fp = open_memmap(q_data_path, dtype='float32', mode='w+',
                shape=(num_sample, 4, max_frame, num_joint, max_body))
    fp[:, 0, ...] = 1.
    # Convert xyz to quaternion and save
    sk = Skeleton(fpha_info.PARENTS)
    xyz = xyz_data.transpose(0, 2, 1, 3)  # (num_samples, num_frame, 3, num_joint)
    for i in tqdm(range(num_sample), ascii=True):
        fp[i][:, :num_frame[i], :, 0] = sk.xyz2qrel(xyz[i][:num_frame[i]]).transpose(1, 0, 2)  # (num_samples, 4, num_frame, num_joint ,1)

if __name__ == '__main__':
    # Args
    parser = argparse.ArgumentParser(description='FPHA Data Converter.')
    parser.add_argument('data_path')
    parser.add_argument('--mode', choices=['xyz','qrel'], default='xyz')

    args = parser.parse_args()

    # Setups
    part = ['train', 'val']
    if args.data_path[-1]=='/': args.data_path = args.data_path[:-1]
    base_dir = args.data_path.rsplit('/', 1)[0]+'/fpha'
    xyz_dir = os.path.join(base_dir, 'xyz')
    qrel_dir = os.path.join(base_dir, 'qrel')

    skeleton_root = os.path.join(args.data_path, 'Hand_pose_annotation_v1')

    # Generate xyz data
    if args.mode == 'xyz': 
        print('Generating xyz data...')
        out_path = xyz_dir
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for p in part:
            print('{}'.format(p))
            # Collect sample names and labels
            sample_name, sample_label = collect_samples(args.data_path, part=p)
            # Read xyz coordinates and save to file
            gen_xyz(skeleton_root, out_path, p, sample_name, sample_label)

    elif args.mode == 'qrel':
        print('Converting to qrel data...')
        out_path = qrel_dir
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for p in part:
            print('{}'.format(p))
            xyz_path = xyz_dir
            # Convert quaternion to qrel
            xyz2qrel(xyz_path, out_path, p)
