import time
import numpy as np
import random
import sys
import os
import argparse
import cv2
import zipfile
import itertools
import pybullet
import json
import time
import numpy as np
import imageio
import pybullet as p

from collect_pose_data import PoseDataCollector
sys.path.insert(1, '../utils/')
from coord_helper import * 
from data_helper import * 
import obj_file

def prepare_one_pose(output_folder_dir, result_file_name, filtered_idx, filtered_obj_pose_seq_arr, object_pc):
	output_file_name_arr = []
	for idx, seqpose in zip(filtered_idx, filtered_obj_pose_seq_arr):
		output_file_name = result_file_name + '-{}'.format(idx)
		output_file_name_arr.append(output_file_name)
		half_output_dir = os.path.join(output_folder_dir, output_file_name)
		startpc_out_dir = half_output_dir + '-startpc.npy' 
		endpc_out_dir = half_output_dir + '-endpc.npy' 
		seqpose_out_dir = half_output_dir + '-seqpose.npy' 
		
		# assert seqpose.shape[0] > 1
		startpc = apply_transform_to_pc_with_n(object_pc, seqpose[0])
		endpc = apply_transform_to_pc_with_n(object_pc, seqpose[-1])
		
		np.save(startpc_out_dir, startpc)
		np.save(endpc_out_dir, endpc)
		np.save(seqpose_out_dir, seqpose)

	return output_file_name_arr


# def filter_one_pose(result_data, result_folder_dir, result_file_name, hook_bullet_id, object_bullet_id, hook_world_pos, hook_scaling, collector):
def process_and_filter_one_pose(result_data, result_folder_dir, result_file_name):
	obj_pos_quat_seq_arr = []
	idx_cutoff_arr = []
	np_dir_arr = []

	for i in range(len(result_data['succ_force'])):
		half_output_dir = os.path.join(result_folder_dir, result_file_name + '-{}'.format(str(i)))
		np_dir = half_output_dir + '-pose.npy'
		assert os.path.isfile(np_dir)
		np_dir_arr.append(np_dir)

	quat_error_arr = []
	filtered_idx = []
	for i in range(len(np_dir_arr)):
		obj_pos_quat_seq = np.load(np_dir_arr[i])
		idx_cutoff_1 = np.searchsorted(np.linalg.norm(obj_pos_quat_seq[:, :3], axis=-1), 0.6) 
		idx_cutoff_2 = np.searchsorted(obj_pos_quat_seq[:, 2], 0.4) 
		idx_cutoff = min(idx_cutoff_1, idx_cutoff_2)
		# print(i, 'idx cutoff', idx_cutoff)
		idx_cutoff_arr.append(idx_cutoff)
		obj_pos_quat_seq_arr.append(obj_pos_quat_seq[:idx_cutoff])	

		# auto disqualify sequences that have <=3 timesteps
		if obj_pos_quat_seq_arr[i].shape[0] <= 3:
			continue
		quat_error = mean_quat_error(obj_pos_quat_seq_arr[i])
		if quat_error < 0.3:
			filtered_idx.append(i)
		quat_error_arr.append([quat_error, i])
	
	if len(quat_error_arr) == 0:
		return [], []
	if len(filtered_idx) == 0:
		quat_error_arr = np.array(quat_error_arr)
		filtered_idx = [int(quat_error_arr[np.argsort(quat_error_arr[:, 0]), 1][0])]

	filtered_idx = np.array(filtered_idx)
	# print(obj_pos_quat_seq_arr[filtered_idx[0]].shape)
	# print(obj_pos_quat_seq_arr[filtered_idx[0]][0])

	# reverse footage
	filtered_obj_pose_seq_arr = [np.flip(obj_pos_quat_seq_arr[i], 0) for i in filtered_idx]
	# print(filtered_obj_pose_seq_arr[0][-1])

	return filtered_idx, filtered_obj_pose_seq_arr
	# print('sort by time', np.argsort(t_arr), np.sort(t_arr))
	# print('sort by quat', np.argsort(quat_error_arr), np.sort(quat_error_arr))
	# print()
	
	


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="../data")
	parser.add_argument("--use_labeled_data", action='store_true')
	parser.add_argument("--sherlock", action='store_true')
	parser.add_argument("--hook_name", default='')
	parser.add_argument("--obj_cat_split_id", type=int, default=-1)

	args = parser.parse_args()

	obj_cat_split_id = int(args.obj_cat_split_id)
	if args.sherlock:
		args.home_dir_data = '/scratch/groups/bohg/hang'
		assert args.hook_name != ''
		assert obj_cat_split_id >= 0
	data_dir = os.path.join(args.home_dir_data, 'geo_data')
	exclude_dir = os.path.join(args.home_dir_data, 'exclude')
	output_dir = os.path.join(args.home_dir_data, 'collection_result')
	collection_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result')
	visualize_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result_visualize')
	chunk_folder_dir = os.path.join(args.home_dir_data, 'geo_data/misc_chunks')
	labeled_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result_labeled')
	seq_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result_seq')
	dataset_folder_dir = os.path.join(args.home_dir_data, 'dataset_seq')
	dataset_labels_folder_dir = os.path.join(args.home_dir_data, 'dataset_seq', 'labels')


	all_hook_name, all_hook_urdf, all_object_name, all_object_urdf = load_all_hooks_object_w_split_id(obj_cat_split_id, data_dir, exclude_dir, None, True, True)


	# p_id = bc.BulletClient(connection_mode=pybullet.GUI)
	# collector = PoseDataCollector(p_id) 

	if not os.path.isdir(dataset_folder_dir):
		os.mkdir(dataset_folder_dir)
	if not os.path.isdir(dataset_labels_folder_dir):
		os.mkdir(dataset_labels_folder_dir)
	ct = 0

	if args.hook_name != '':
		assert args.hook_name in all_hook_name

	for i, hook_name in enumerate(all_hook_name):
		if args.hook_name != '' and args.hook_name != hook_name:
			continue
	# for visualize_labeled_folder_name in os.listdir(labeled_result_folder_dir):
		# hook_name = visualize_labeled_folder_name.replace('visualize_chunk_', '')
		# i = all_hook_name.index(hook_name)
		# if not hook_name == 'hook_wall_124':
		# if not hook_name == 'hook_wall_185':
		# if not hook_name == 'hook_wall_75':
			# continue
		# hook_urdf_dir = all_hook_urdf[i]
		# hook_pc_dir = get_numpy_dir_from_urdf(hook_urdf_dir)
		# hook_pc = np.load(hook_pc_dir)

		# hook_bullet_id, hook_scaling = collector.init_hook(hook_urdf_dir)
		# hook_world_pos_offset = get_hook_wall_offset(hook_urdf_dir)
		# hook_world_pos = collector.get_hook_world_pos(hook_bullet_id, hook_world_pos_offset)
		# print('hook world pos offest', hook_world_pos_offset)
		
		output_file_name_arr = []
		for j, object_name in enumerate(all_object_name):
			# if not 'mug' in object_name:
				# continue
			# if int(object_name.split('_')[-1]) < 23:
				# continue
			# print(object_name)
			object_urdf_dir = all_object_urdf[j]
			object_pc_dir = get_numpy_dir_from_urdf(object_urdf_dir)
			object_pc = np.load(object_pc_dir)
			result_file_name = hook_name + '_' + object_name
			result_file_dir = os.path.join(collection_result_folder_dir, result_file_name + '.txt')
			if not os.path.isfile(result_file_dir):
				continue
			result_np = load_result_file(result_file_dir)
			excluded_rows = []
			if args.use_labeled_data:
				for k in range(result_np.shape[0]):
					image_dir = os.path.join(labeled_result_folder_dir, 'visualize_chunk_{}'.format(hook_name), '{}_{}.jpg'.format(result_file_name, str(k)))
					if not os.path.isfile(image_dir):
						excluded_rows.append(k)
			if len(excluded_rows) == result_np.shape[0]:
				continue

			# print(hook_pc_dir, object_pc_dir)
			# print(hook_urdf_dir, object_urdf_dir)
			ct += 1
			# object_bullet_id = collector.p.loadURDF(object_urdf_dir, basePosition=[0, 0, 2], baseOrientation=[0, 0, 0, 1], globalScaling=1, useFixedBase=False)
			# object_scaling = collector.p.getCollisionShapeData(object_bullet_id, -1)[0][3][0] 	
			info_output_file_dir = os.path.join(seq_result_folder_dir, '{}.json'.format(result_file_name))
			info_output_arr = []

			print(result_file_name)
			for k in range(result_np.shape[0]):
				if k in excluded_rows:
					continue
				object_pos_local = result_np[k, -7:-4]
				object_quat_local = result_np[k, -4:]
				k_result_file_name = '{}-{}'.format(result_file_name, str(k))
				result_json_dir =  os.path.join(seq_result_folder_dir, k_result_file_name + '.json')

				with open(result_json_dir) as f:
					result_data = json.load(f)
				if len(result_data['succ_force']) == 0:
					continue

				# process_and_filter_one_pose(result_data, seq_result_folder_dir, k_result_file_name,  hook_bullet_id, object_bullet_id, hook_world_pos, hook_scaling, collector)
				# filter_one_pose(result_data, seq_result_folder_dir, k_result_file_name)
				filtered_idx, filtered_obj_pose_seq_arr = process_and_filter_one_pose(result_data, seq_result_folder_dir, k_result_file_name)
				if len(filtered_idx) == 0:
					continue
				
				tmp_output_file_name_arr = prepare_one_pose(dataset_folder_dir, k_result_file_name, filtered_idx, filtered_obj_pose_seq_arr, object_pc)
				output_file_name_arr += tmp_output_file_name_arr
		labels_out_dir = os.path.join(dataset_labels_folder_dir, '{}.txt'.format(hook_name))

		with open(labels_out_dir, 'w+') as f:
			for line in output_file_name_arr:
				f.write(line + '\n')

		

	