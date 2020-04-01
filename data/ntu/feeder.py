# sys
import numpy as np
import pickle

# torch
import torch
import torch.utils.data

from data.ntu.ntu_info import LABEL_NAMES
class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        num_samples: (int) number of samples selected from the whole dataset. if -1, use all
        num_frames: (int) number of frames
        seed: (int) random seed
    """

    def __init__(self, data_path, label_path, bone_path=None, num_samples=-1, mmap=True, num_frames=300, seed=0):
        np.random.seed(seed)
        self.data_path = data_path
        self.bone_path = bone_path
        self.label_path = label_path
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.load_data(mmap)

    def load_data(self, mmap):

        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label, self.num_frames_data = pickle.load(f)
        
        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')  # (N, C, T, V, M)
        else:
            self.data = np.load(self.data_path)
        
        if self.bone_path is not None:
            if mmap:
                self.data_bone = np.load(self.bone_path, mmap_mode='r')
            else:
                self.data_bone = np.load(self.bone_path)
                
        # Sub sample. Randomly sample in each interval
        self.ind = []
        random_sample = False
        for i in range(len(self.label)):
            interval = np.max([self.num_frames_data[i] // self.num_frames, 1])
            if random_sample:
                self.ind.append([np.random.randint(interval) + interval * i for i in range(self.num_frames)])
            else:
                self.ind.append([interval * i for i in range(self.num_frames)])

        if self.num_samples is not -1:
            ind = np.random.choice(len(self.label), size=self.num_samples, replace=False)
            self.label = np.array(self.label).take(ind, 0)
            self.data = self.data.take(ind, 0)
            self.sample_name = np.array(self.sample_name).take(ind, 0)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index].take(self.ind[index], axis=1))  # C, T, V, M
        label = self.label[index]
        if self.bone_path is not None:
            data_bone_numpy = np.array(self.data_bone[index].take(self.ind[index], axis=1))  # C, T, V, M       
            return (data_numpy, data_bone_numpy), label
        else:
            return data_numpy, label

