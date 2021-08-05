
import os
import os.path
import json
import numpy as np
import sys
import itertools
import random
import argparse
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'utils'))
sys.path.append(UTILS_DIR)
sys.path.append(BASE_DIR)
# sys.path.insert(1, '../utils/')
from data_helper import * 
from coord_helper import *


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="../data")
	parser.add_argument("--use_labeled_data", action='store_true')
	parser.add_argument("--hook_name", default='')
	parser.add_argument("--obj_cat_split_id", type=int, default=-1)
	args = parser.parse_args()
	
	args.use_labeled_data = True
	obj_cat_split_id = int(args.obj_cat_split_id)
	home_dir_data = args.home_dir_data
	data_dir = os.path.join(args.home_dir_data, 'geo_data')
	pose_result_folder_dir = os.path.join(home_dir_data, 'collection_result')
	exclude_dir = os.path.join(args.home_dir_data, 'exclude')
	dataset_folder_dir = os.path.join(home_dir_data, 'dataset_pose')
	dataset_labels_folder_dir = os.path.join(args.home_dir_data, 'dataset_pose', 'labels')

	labeled_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result_labeled')

	all_hook_name, all_hook_urdf, all_object_name, all_object_urdf = load_all_hooks_object_w_split_id(obj_cat_split_id, data_dir, exclude_dir, None, True, True)

	if not os.path.isdir(dataset_folder_dir):
		os.mkdir(dataset_folder_dir)
	if not os.path.isdir(dataset_labels_folder_dir):
		os.mkdir(dataset_labels_folder_dir)

	if args.hook_name != '':
		assert args.hook_name in all_hook_name

	for i, hook_name in enumerate(all_hook_name):
		if args.hook_name != '' and args.hook_name != hook_name:
			continue

		output_file_name_arr = []
		for j, object_name in enumerate(all_object_name):
			object_urdf_dir = all_object_urdf[j]
			object_pc_dir = get_numpy_dir_from_urdf(object_urdf_dir)
			object_pc = np.load(object_pc_dir)
			result_file_name = hook_name + '_' + object_name
			result_file_dir = os.path.join(pose_result_folder_dir, result_file_name + '.txt')
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
			for k in range(result_np.shape[0]):
				if k in excluded_rows:
					continue
				k_result_file_name = '{}-{}'.format(result_file_name, str(k))

				# export ending point cloud
				endpc = apply_transform_to_pc_with_n(object_pc, result_np[k, -7:])
				endpc_out_dir = os.path.join(dataset_folder_dir, k_result_file_name + '-endpc.npy') 

				# print(endpc_out_dir)
				np.save(endpc_out_dir, endpc)
				print(endpc_out_dir)
				
				output_file_name_arr.append(k_result_file_name)

		if len(output_file_name_arr):
			labels_out_dir = os.path.join(dataset_labels_folder_dir, '{}.txt'.format(hook_name))
			print(labels_out_dir)
			with open(labels_out_dir, 'w+') as f:
				for line in output_file_name_arr:
					f.write(line + '\n')

	