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
		# assert args.hook_name != ''
		# assert obj_cat_split_id >= 0
	data_dir = os.path.join(args.home_dir_data, 'geo_data')
	exclude_dir = os.path.join(args.home_dir_data, 'exclude')
	output_dir = os.path.join(args.home_dir_data, 'collection_result')
	collection_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result')
	visualize_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result_visualize')
	chunk_folder_dir = os.path.join(args.home_dir_data, 'geo_data/misc_chunks')
	labeled_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result_labeled')
	seq_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result_seq')


	all_hook_name, all_hook_urdf, all_object_name, all_object_urdf = load_all_hooks_object_w_split_id(obj_cat_split_id, data_dir, exclude_dir, None, True, True)

	ct = 0

	if args.hook_name != '':
		assert args.hook_name in all_hook_name

	all_k_result_file_name_dict = {}
	all_k_ct = 0
	all_k_success_ct = 0
	error_ct = 0
	for i, hook_name in enumerate(all_hook_name):
		tmp1 = 0
		tmp2 = 0
		if args.hook_name != '' and args.hook_name != hook_name:
			continue
		# if os.path.isdir(os.path.join(labeled_result_folder_dir, 'visualize_chunk_{}'.format(hook_name))):
		# 	if len(os.listdir(os.path.join(labeled_result_folder_dir, 'visualize_chunk_{}'.format(hook_name)))) > 0:
		# 		print('skip', hook_name)
		# 		continue
		output_file_name_arr = []
		for j, object_name in enumerate(all_object_name):
			# if not 'mug' in object_name:
				# continue
			# if int(object_name.split('_')[-1]) < 23:
				# continue
			# print(object_name)
			object_urdf_dir = all_object_urdf[j]
			object_pc_dir = get_numpy_dir_from_urdf(object_urdf_dir)
			# object_pc = np.load(object_pc_dir)
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

			ct += 1
			info_output_file_dir = os.path.join(seq_result_folder_dir, '{}.json'.format(result_file_name))
			info_output_arr = []

			# print(result_file_name)
			for k in range(result_np.shape[0]):
				if k in excluded_rows:
					continue
				object_pos_local = result_np[k, -7:-4]
				object_quat_local = result_np[k, -4:]
				k_result_file_name = '{}-{}'.format(result_file_name, str(k))
				result_json_dir =  os.path.join(seq_result_folder_dir, k_result_file_name + '.json')
				if not os.path.isfile(result_json_dir):
					error_ct += 1
					continue
				with open(result_json_dir) as f:
					result_data = json.load(f)
				all_k_ct += 1
				tmp1 += 1
				if len(result_data['succ_force']) == 0:
					#print(result_json_dir)
					continue

				all_k_result_file_name_dict[k_result_file_name] = 1
				all_k_success_ct += 1
				tmp2 += 1
		print(tmp2, tmp1, hook_name)
		
	print(all_k_ct, all_k_success_ct, error_ct)
				
				# tmp_output_file_name_arr = prepare_one_pose(dataset_folder_dir, k_result_file_name, filtered_idx, filtered_obj_pose_seq_arr, object_pc)
				# output_file_name_arr += tmp_output_file_name_arr
		# labels_out_dir = os.path.join(dataset_labels_folder_dir, '{}.txt'.format(hook_name))

		# with open(labels_out_dir, 'w+') as f:
			# for line in output_file_name_arr:
				# f.write(line + '\n')

		

	
