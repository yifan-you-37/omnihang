
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
	parser.add_argument("--hook_name", default='')
	parser.add_argument("--obj_cat_split_id", type=int, default=-1)
	args = parser.parse_args()

	obj_cat_split_id = int(args.obj_cat_split_id)
	home_dir_data = args.home_dir_data
	labels_folder_dir = os.path.join(args.home_dir_data, 'geo_data/labels/')
	cp_result_folder_dir = os.path.join(home_dir_data, 'dataset_cp')
	cp_mat_folder_dir = os.path.join(home_dir_data, 'dataset_cp_mat')
	pose_result_folder_dir = os.path.join(home_dir_data, 'collection_result')
	seq_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result_seq')
	labeled_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result_labeled')
	exclude_dir = os.path.join(args.home_dir_data, 'exclude')
	data_dir = os.path.join(args.home_dir_data, 'geo_data')
	collection_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result')

	all_data = {}
	ct = 0

	all_hook_name, all_hook_urdf, all_object_name, all_object_urdf = load_all_hooks_object_w_split_id(obj_cat_split_id, data_dir, exclude_dir, labels_folder_dir, True, True)

	all_data = {}
	for i, hook_name in enumerate(all_hook_name):
		if args.hook_name != '' and args.hook_name != hook_name:
			continue
		for j, object_name in enumerate(all_object_name):
			result_file_name = hook_name + '_' + object_name
			pose_result_file_dir = os.path.join(collection_result_folder_dir, result_file_name + '.txt')

			if not os.path.exists(pose_result_file_dir):
				continue

			result_file_dir = os.path.join(cp_result_folder_dir, result_file_name + '.json')
			if not os.path.exists(result_file_dir):
				continue	
			try:
				with open(result_file_dir) as f:
					tmp = json.load(f)
			except:
				os.remove(result_file_dir)
				continue
			#b = time.time()
			final_arr = []
			for tmp_one in tmp:
				# result_json_dir = os.path.join(seq_result_folder_dir, result_file_name + '-' + str(k) + '.json')
				k = tmp_one['index']
				np_dir = os.path.join(seq_result_folder_dir, result_file_name + '-' + str(k) + '-0-pose.npy')
				if os.path.exists(np_dir):
					final_arr.append(tmp_one)

			ct += 1
			if len(final_arr) > 0:
				with open(result_file_dir, 'w+') as f:
					json.dump(final_arr, f)
				print(result_file_dir)
			else:
				os.remove(result_file_dir)
	
	
