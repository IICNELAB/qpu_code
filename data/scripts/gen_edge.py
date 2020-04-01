import os
import sys
sys.path.append('.')

import numpy as np
from numpy.lib.format import open_memmap
import argparse

from utils.quaternion import q_conj, q_mul

paris = {
    'ntu': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 12),
        (25, 12)
    ),
    'fpha':(
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 1), (7, 6), (8, 7), (9, 2), (10, 9), (11, 10), (12, 3), 
        (13, 12), (14, 13), (15, 4), (16, 15), (17, 16), (18, 5), (19, 18), (20, 19)
    )
}

# bone
from tqdm import tqdm

def gen_edge(root,dataset):
    for part in ['train','val']:
        print(part)
        data = np.load(os.path.join(root,'{}_data.npy'.format(part)))
        N, C, T, V, M = data.shape
        fp_sp = open_memmap(
            os.path.join(root,'{}_data_rel.npy'.format(part)),
            dtype='float32',
            mode='w+',
            shape=(N, 4, T, V, M))

        fp_sp[:, :C, :, :, :] = data
        data = data.transpose(0, 2, 3, 4, 1)  # N, T, V, M, C
        for v1, v2 in tqdm(paris[dataset],ascii=True):
            v1 -= 1
            v2 -= 1
            fp_sp[:, :, :, v1, :] = q_mul(data[:, :, v1, :, :], q_conj(data[:, :, v2, :, :])).transpose(0, 3, 1, 2)  # N, C, T, M

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bone (edge) data Converter.')
    parser.add_argument('data_path')
    parser.add_argument('--dataset',choices=['ntu', 'fpha'],default='ntu')
    args = parser.parse_args()

    gen_edge(args.data_path,args.dataset)
