import pybullet 
import time
import sys
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
# sys.path.insert(1, '../simulation/')
# from Microwave_Env import RobotEnv
from collect_pose_data import PoseDataCollector
sys.path.insert(1, '../lin/')
from classifier_dataset_torch import ClassifierDataset
sys.path.insert(1, '../utils/')
from coord_helper import * 
from data_helper import * 
import obj_file
import json
import bullet_client as bc
from scipy.spatial import KDTree

try:
	from mayavi import mlab as mayalab
except:
	pass
simulation_dir = '../simulation/'
sys.path.insert(0, simulation_dir)
from Microwave_Env import RobotEnv


def get_partial_idx(cp_idx, partial_pc_idx):
	partial_pc_idx_dict = {}

	for i in range(partial_pc_idx.shape[0]):
		idx = partial_pc_idx[i]
		if idx in partial_pc_idx_dict:
			partial_pc_idx_dict[idx].append(i)
		else:
			partial_pc_idx_dict[idx] = [i]
	
	partial_idx = []

	overlap_ct = 0
	for one_cp_idx in cp_idx:
		if one_cp_idx in partial_pc_idx_dict:
			partial_idx += partial_pc_idx_dict[one_cp_idx]
			overlap_ct += 1
	return np.array(partial_idx), overlap_ct * 1./(len(cp_idx) + 0.01)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="../data")
	parser.add_argument("--hook_name", default='')
	parser.add_argument("--object_cat", default='')
	parser.add_argument("--start_id", type=int)
	parser.add_argument("--obj_cat_split_id", type=int, default=-1)
	args = parser.parse_args()

	obj_cat_split_id = int(args.obj_cat_split_id)
	data_dir = os.path.join(args.home_dir_data, 'geo_data')
	labels_folder_dir = os.path.join(args.home_dir_data, 'geo_data/labels/')
	exclude_dir = os.path.join(args.home_dir_data, 'exclude')
	cp_result_folder_dir = os.path.join(args.home_dir_data, 'dataset_cp')

	partial_pc_folder = os.path.join(args.home_dir_data, 'geo_data_partial_cp_pad')

	train_list_dir = os.path.join(cp_result_folder_dir, 'labels', 'train_list.txt')
	test_list_dir = os.path.join(cp_result_folder_dir, 'labels', 'test_list.txt')

	train_set_one_per_pair = ClassifierDataset(args.home_dir_data, train_list_dir, False, split='train', with_wall=False, one_per_pair=False)
	test_set_one_per_pair = ClassifierDataset(args.home_dir_data, test_list_dir, False, split='test', with_wall=False, one_per_pair=False)

	ct = 0

	overlap_dict = {}

	out_partial_folder = os.path.join(args.home_dir_data, 'dataset_cp_partial')
	mkdir_if_not(out_partial_folder)

	all_overlap_o =[]
	all_overlap_h = []

	for dataset in [train_set_one_per_pair, test_set_one_per_pair]:
		for i, batch_dict in enumerate(dataset.all_data):

			result_file_name = batch_dict['result_file_name']
			pose_idx = int(batch_dict['pose_idx'])
			cp_result_file_name = result_file_name + '_' + str(pose_idx)

			hook_name = batch_dict['hook_name']
			if hook_name != args.hook_name and args.hook_name != '':
				continue
			object_name = batch_dict['object_name']

			hook_urdf = train_set_one_per_pair.all_hook_dict[hook_name]['urdf']
			hook_pc_dir = train_set_one_per_pair.all_hook_dict[hook_name]['pc']

			object_urdf = train_set_one_per_pair.all_object_dict[object_name]['urdf']
			object_pc_dir = train_set_one_per_pair.all_object_dict[object_name]['pc']

			cp_idx_o_dir = os.path.join(cp_result_folder_dir, cp_result_file_name +  '_idx_object.npy')
			cp_idx_h_dir = os.path.join(cp_result_folder_dir, cp_result_file_name +  '_idx_hook.npy')

			partial_idx_o_dir = os.path.join(out_partial_folder, cp_result_file_name +  '_idx_object.npy')
			partial_idx_h_dir = os.path.join(out_partial_folder, cp_result_file_name +  '_idx_hook.npy')

			partial_pc_idx_o_dir = os.path.join(partial_pc_folder, result_file_name + '_object_partial_pc_pad_idx.npy')
			partial_pc_idx_h_dir = os.path.join(partial_pc_folder, result_file_name + '_hook_partial_pc_pad_idx.npy')

			partial_pc_o_dir = os.path.join(partial_pc_folder, result_file_name + '_object_partial_pc_pad.npy')
			partial_pc_h_dir = os.path.join(partial_pc_folder, result_file_name + '_hook_partial_pc_pad.npy')


			if (not os.path.exists(cp_idx_o_dir)) or (not os.path.exists(cp_idx_h_dir)):
				continue 
			cp_idx_o = np.load(cp_idx_o_dir)
			cp_idx_h = np.load(cp_idx_h_dir)

			partial_pc_idx_o = np.load(partial_pc_idx_o_dir)
			partial_pc_idx_h = np.load(partial_pc_idx_h_dir)
			# if partial_pc_idx_o.shape[0] == 0:
				# print(partial_pc_idx_o_dir)
			# if partial_pc_idx_h.shape[0] == 0:
				# print(partial_pc_idx_h_dir)
			partial_idx_o, tmp_overlap_o = get_partial_idx(cp_idx_o, partial_pc_idx_o)
			partial_idx_h, tmp_overlap_h = get_partial_idx(cp_idx_h, partial_pc_idx_h)
			all_overlap_o.append(tmp_overlap_o)
			all_overlap_h.append(tmp_overlap_h)

			if i % 500 == 0:
				print(args.hook_name,'o overlap', np.mean(all_overlap_o))
				print(args.hook_name,'h overlap', np.mean(all_overlap_h))

			np.save(partial_idx_o_dir, partial_idx_o)
			np.save(partial_idx_h_dir, partial_idx_h)


			# hook_pc = np.load(hook_pc_dir)
			# object_pc = np.load(object_pc_dir)
			# visualize_pc_n(object_pc)
			# visualize_pc_n(object_pc[cp_idx_o], color=[1, 0, 0])
			# mayalab.show()

			# partial_pc_o = np.load(partial_pc_o_dir)
			# partial_pc_h = np.load(partial_pc_h_dir)
			# # assert np.allclose(np.linalg.norm(partial_pc_o[:, -3:], axis=1))
			# assert np.allclose(np.linalg.norm(partial_pc_o[partial_idx_o][:, -3:], axis=1), np.ones_like(np.linalg.norm(partial_pc_o[partial_idx_o][:, -3:], axis=1)))
			# visualize_pc_n(partial_pc_o)
			# visualize_pc_n(partial_pc_o[partial_idx_o], color=[1, 0, 0])
			# mayalab.show()
			# break
