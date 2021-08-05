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

# !! command 79 of 85 ("python collect_pose_data_pene_no_wall.py --home_dir_data /juno/downloads/new_hang --hook_name hook_wall_horiz_rod_1") failed
# !! command 84 of 85 ("python collect_pose_data_pene_no_wall.py --home_dir_data /juno/downloads/new_hang --hook_name hook_wall_horiz_rod_39") failed

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="/home/yifanyou/hang")
	args = parser.parse_args()

	data_dir = os.path.join(args.home_dir_data, 'geo_data')
	labels_folder_dir = os.path.join(args.home_dir_data, 'geo_data/labels/')
	exclude_dir = os.path.join(args.home_dir_data, 'exclude')

	partial_pc_folder = os.path.join(args.home_dir_data, 'geo_data_partial_cp')
	out_pc_folder = os.path.join(args.home_dir_data, 'geo_data_partial_cp_pad')
	mkdir_if_not(out_pc_folder)

	# partial_pc_labels_folder = os.path.join(partial_pc_folder, 'labels')
	# all_overlap_dict = {}
	# for labels_file in os.listdir(partial_pc_labels_folder):
	# 	with open(os.path.join(partial_pc_labels_folder, labels_file)) as f:
	# 		tmp = json.load(f)
	# 	all_overlap_dict.update(tmp)
	# print(len(all_overlap_dict))

	n_pc = 4096

	for partial_pc_name in os.listdir(partial_pc_folder):
		if not (partial_pc_name.endswith('partial_pc.npy')):
			continue
			
		partial_pc_dir = os.path.join(partial_pc_folder, partial_pc_name)

		partial_pc_idx_name = partial_pc_name[:-4] + '_idx.npy'
		result_file_name = partial_pc_name[:-4]
		hook_cat, hook_id, object_cat, object_id = decode_result_file_name(result_file_name)
		# if 'hook_partial_pc' in partial_pc_name:
		# 	ori_pc = np.load(os.path.join(data_dir, hook_cat, str(hook_id), 'model_meshlabserver_normalized_pc.npy'))
		# else:
		# 	ori_pc_dir = os.path.join(data_dir, object_cat, str(object_id), 'model_normalized_v_pc.npy')
		# 	if not os.path.exists(ori_pc_dir):
		# 		ori_pc_dir = os.path.join(data_dir, object_cat, str(object_id), 'model_meshlabserver_normalized_v_pc.npy')
		# 	ori_pc = np.load(ori_pc_dir)

		partial_pc_idx_dir = os.path.join(partial_pc_folder, partial_pc_idx_name)
		if not os.path.isfile(partial_pc_dir):
			continue

		out_pc_name = partial_pc_name[:-4] + '_pad.npy'
		out_pc_dir = os.path.join(out_pc_folder, out_pc_name)
		out_idx_dir = os.path.join(out_pc_folder, partial_pc_name[:-4] + '_pad_idx.npy')


		partial_pc = np.load(partial_pc_dir)
		partial_pc_idx = np.load(partial_pc_idx_dir)

		print(partial_pc_name, partial_pc.shape)
		assert partial_pc.shape[1] == 6
		n_partial_pc = partial_pc.shape[0]
		n_sample = n_pc - n_partial_pc

		full_pc = np.zeros((n_pc, 6))
		full_pc[:n_partial_pc] = partial_pc

		sample_idx = np.random.choice(n_partial_pc, n_sample)
		full_pc[n_partial_pc:] = partial_pc[sample_idx, :]

		all_idx = np.append(partial_pc_idx, partial_pc_idx[sample_idx])

		
		# assert np.allclose(ori_pc[all_idx], full_pc)

		# obj_pc = full_pc
		# from mayavi import mlab as mayalab 
		# # plot_pc(obj_pc)
		# mayalab.quiver3d(obj_pc[:, 0], obj_pc[:, 1], obj_pc[:, 2], obj_pc[:, 3], obj_pc[:, 4], obj_pc[:, 5], scale_factor=0.005)
		# # # if data_dict['label'] == 1:
		# # print('label', data_dict['label'], data_dict['object_name'], data_dict['hook_name'])
		# # print(pose_o_end, obj_pc.shape)
		# # # mayalab.points3d(obj_pc[:, 0], obj_pc[:, 1], obj_pc[:, 2], scale_factor=0.005)
		# # print('show it')
		# mayalab.show()

		# obj_pc = partial_pc
		# # plot_pc(partial_pc, color=[1, 0, 0])
		# mayalab.quiver3d(obj_pc[:, 0], obj_pc[:, 1], obj_pc[:, 2], obj_pc[:, 3], obj_pc[:, 4], obj_pc[:, 5], scale_factor=0.005)
		# mayalab.show()

		# print(out_pc_dir, full_pc.shape)
		np.save(out_pc_dir, full_pc)
		np.save(out_idx_dir, all_idx)
		
		

 