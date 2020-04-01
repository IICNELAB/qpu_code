import os
import sys
sys.path.append('.')

import numpy as np

from utils.skel_utils.skeleton import Skeleton


def ntu_avg_bone_length():
	from data.ntu.feeder import Feeder
	from data.ntu import ntu_info

	train_data_path = '/media/xuan/ssd/data/NTU-RGB-D/xyz/xsub/train_data.npy'
	train_label_path = '/media/xuan/ssd/data/NTU-RGB-D/xyz/xsub/train_label.pkl'
	train_data = Feeder(train_data_path, train_label_path, num_samples=-1, num_frames=300, mmap=True)
	xyz = np.zeros((3, 25))
	total_frames = 0
	for i in range(len(train_data)):
		seq, _ = train_data[i]
		for j in range(train_data.num_frames_data[i]):
			xyz += seq[:, j, :]
		total_frames += train_data.num_frames_data[i]
	xyz /= total_frames
	sk = Skeleton(parents=ntu_info.PARENTS)
	print(sk.compute_bone_lens(xyz))


def fpha_avg_bone_length():
	from data.fpha.feeder import Feeder
	from data.fpha import fpha_info

	train_data_path = '/media/xuan/ssd/data/fpha_processed/xyz/train_data.npy'
	train_label_path = '/media/xuan/ssd/data/fpha_processed/xyz/train_label.pkl'
	train_data = Feeder(train_data_path, train_label_path, num_samples=-1, num_frames=300, mmap=True)
	xyz = np.zeros((3, 21))
	total_frames = 0
	for i in range(len(train_data)):
		seq, _ = train_data[i]
		for j in range(train_data.num_frames_data[i]):
			xyz += seq[:, j, :]
		total_frames += train_data.num_frames_data[i]
	xyz /= total_frames
	sk = Skeleton(parents=fpha_info.PARENTS)
	print(sk.compute_bone_lens(xyz))


def dhg_avg_bone_length():
	from data.dhg.feeder import Feeder
	from data.dhg import dhg_info

	train_data_path = '/media/xuan/ssd/data/dhg_processed/xyz/train_data.npy'
	train_label_path = '/media/xuan/ssd/data/dhg_processed/xyz/train_14_label.pkl'
	train_data = Feeder(train_data_path, train_label_path, num_samples=-1, num_frames=200, mmap=True)
	xyz = np.zeros((3, 22))
	total_frames = 0
	for i in range(len(train_data)):
		seq, _ = train_data[i]
		for j in range(train_data.num_frames_data[i]):
			xyz += seq[:, j, :]
		total_frames += train_data.num_frames_data[i]
	xyz /= total_frames
	sk = Skeleton(parents=dhg_info.PARENTS)
	print(sk.compute_bone_lens(xyz))

if __name__ == '__main__':
	# ntu_avg_bone_length()
	fpha_avg_bone_length()
	# dhg_avg_bone_length()