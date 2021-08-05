
import pybullet 
import time
import numpy as np
import random
import sys
import os
import argparse

from collect_pose_data import PoseDataCollector

sys.path.insert(1, '../utils/')
from coord_helper import * 
from data_helper import * 
import bullet_client as bc


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="/home/yifany/")
	parser.add_argument('--hook_category', default=None)
	parser.add_argument('--object_category', default=None)
	parser.add_argument('--use_collection_visualize', action='store_true')
	parser.add_argument('--sherlock', action="store_true")

	args = parser.parse_args()

	if args.sherlock:
		args.home_dir_data = '/scratch/groups/bohg/hang'

	data_dir = os.path.join(args.home_dir_data, 'geo_data')
	labels_folder_dir = os.path.join(args.home_dir_data, 'geo_data/labels/')
	exclude_dir = os.path.join(args.home_dir_data, 'exclude')
	output_dir = os.path.join(args.home_dir_data, 'collection_result')
	collection_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result')
	visualize_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result_visualize')
	chunk_folder_dir = os.path.join(args.home_dir_data, 'geo_data/misc_chunks')
	labeled_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result_labeled')
	zip_folder_dir = os.path.join(args.home_dir_data, 'collection_result_visualize_zip')


	all_hook_name, all_object_name = load_all_hooks_objects(data_dir, exclude_dir, labels_folder_dir)

	print('num hooks', len(all_hook_name))
	print('num objects', len(all_object_name))

	n_pair_total = 0
	n_pair_success = 0
	n_pair_error = 0
	if args.use_collection_visualize:
		for result_folder in os.listdir(visualize_result_folder_dir):
			hook_category, hook_id, object_category, object_id = decode_result_file_name(result_folder)
			hook_name = '{}_{}'.format(hook_category, int(hook_id))
			object_name = '{}_{}'.format(object_category, int(object_id))

			if not(hook_name in all_hook_name):
				continue
			if not(object_name in all_object_name):
				continue
			n_pair_total += 1
			result_folder_dir = os.path.join(visualize_result_folder_dir, result_folder)
			if len(os.listdir(result_folder_dir)) > 0:
				n_pair_success += 1
	else:
		for result_file in os.listdir(collection_result_folder_dir):
			n_pair_total += 1
			hook_category, hook_id, object_category, object_id = decode_result_file_name(result_file[:-4])
			hook_name = '{}_{}'.format(hook_category, int(hook_id))
			object_name = '{}_{}'.format(object_category, int(object_id))

			if not(hook_name in all_hook_name):
				continue
			if not(object_name in all_object_name):
				continue
			result_file_dir = os.path.join(collection_result_folder_dir, result_file)
			if os.path.getsize(result_file_dir) > 0:
				n_pair_success += 1	
				result_folder_dir = os.path.join(visualize_result_folder_dir, result_file[:-4])
				# if not os.path.isdir(result_folder_dir):
					# print(result_folder_dir)
					# print(result_file_dir)
					# n_pair_error += 1
					# print(n_pair_error)
			else:
				pass
	print('total pairs loaded', n_pair_total)
	print('success pairs loaded', n_pair_success)