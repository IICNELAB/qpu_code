import os
import sys
sys.path.append('.')

import pickle
import argparse
import numpy as np
from numpy.lib.format import open_memmap

from data.ntu.ntu_read_skeleton import read_xyz
from tqdm import tqdm
import random
from shutil import copyfile

from utils.skel_utils.skeleton import Skeleton
from utils.quaternion import q_angle
from data.ntu import ntu_info


# Dataset settings
training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
MAX_BODY = 4  # Select 2 bodies from first 4
MAX_BODY_TRUE = 2
NUM_JOINT = 25
MAX_FRAME = 300


def collect_samples(data_path, ignored_sample_path=None, benchmark='xview', part='val', sub_samples=None):
    """Collect selected sample names and labels.
        data_path: (str) directory containing skeleton files
    Args: 
        ignored_sample_path: (str) skeleton files to be ignored
        sub_samples: (int) number of samples to be subsampled. None if no subsampling.
        benchmark: (str) 'xview' or 'xsub'
        part: (str) 'val' or 'train' 
    return:
        sample_name: (list(str)) list of selected sample names
        sample_label: (list(int) list of selected sample labels
    """
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [line.strip() + '.skeleton' for line in f.readlines()]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
        if filename in ignored_samples:
            continue
        action_class = int(filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(filename[filename.find('C') + 1:filename.find('C') + 4])

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)
   
    # random sub sample
    if sub_samples is not None:
        rand_ind = random.sample(range(len(sample_name)), k=sub_samples)
        sample_name = [sample_name[i] for i in rand_ind]
        sample_label = [sample_label[i] for i in rand_ind]

    return sample_name, sample_label


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
                    shape=(len(sample_label), 3, MAX_FRAME, NUM_JOINT, MAX_BODY_TRUE))
    num_frame_list = []
    # Process and save joint data
    for i, s in tqdm(enumerate(sample_name), total=len(sample_name)):
        data, num_frame = read_xyz(os.path.join(data_path, s), max_body=MAX_BODY, 
                                    max_body_true=MAX_BODY_TRUE, num_joint=NUM_JOINT)
        fp[i, :, :data.shape[1], :, :] = data
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
    # Read xyz data (num_samples, 3, num_frame, num_joint, max_body)
    xyz_data = np.load(xyz_data_path, mmap_mode='r')  
    num_sample, _, max_frame, num_joint, max_body = xyz_data.shape
    # Get num_frame
    with open(q_label_path, 'rb') as f:
        _, _, num_frame = pickle.load(f)
    # Open memory to write
    fp = open_memmap(q_data_path, dtype='float32', mode='w+',
                shape=(num_sample, 4, max_frame, num_joint, max_body))
    fp[:, 0, ...] = 1.
    # Convert xyz to quaternion and save
    sk = Skeleton(ntu_info.PARENTS)
    xyz = xyz_data.transpose(0, 2, 4, 1, 3)  # (num_samples, num_frame, max_body, 3, num_joint)
    for i in tqdm(range(num_sample), ascii=True):
        fp[i][:, :num_frame[i], :, :] = sk.xyz2qrel(xyz[i][:num_frame[i]]).transpose(2, 0, 3, 1)  # (num_samples, 4, num_frame, num_joint, max_body)


if __name__ == '__main__':
    # Args
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument('data_path', help='End with nturgb+d_skeletons')
    parser.add_argument('--mode',choices=['xyz','qrel'], default='xyz')
    parser.add_argument('--bench',choices=['xview','xsub'], action = 'append', default=[])
    parser.add_argument('--ignored_sample_path', default='./data/ntu/samples_with_missing_skeletons.txt')

    args = parser.parse_args()

    # Setups
    benchmark = ['xview'] if len(args.bench)==0 else args.bench
    part = ['train', 'val']
    if args.data_path[-1]=='/': args.data_path = args.data_path[:-1]
    base_dir = args.data_path.rsplit('/', 1)[0]+'/NTU-RGB-D'
    sub_samples=None
    xyz_dir = os.path.join(base_dir, 'xyz')
    qrel_dir = os.path.join(base_dir, 'qrel')

    # Generate xyz data
    if args.mode == 'xyz': 
        print('Generating xyz data...')
        for b in benchmark:
            out_path = os.path.join(xyz_dir, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            for p in part:
                print('{}, {}'.format(b, p))
                # Collect sample names and labels
                sample_name, sample_label = collect_samples(args.data_path, args.ignored_sample_path, benchmark=b, part=p, sub_samples=sub_samples)
                # Read xyz coordinates and save to file
                gen_xyz(args.data_path, out_path, p, sample_name, sample_label)

   # Convert to quaternion abs angle. Must ensure xyz data is already generated
    elif args.mode == 'qrel':
        for b in benchmark:
            out_path = os.path.join(qrel_dir, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            for p in part:
                print('{}, {}'.format(b, p))
                xyz_path = os.path.join(xyz_dir, b)
                # Convert quaternion to qrel
                xyz2qrel(xyz_path, out_path, p)

