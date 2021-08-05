import time
import numpy as np
import random
import sys
import os
import argparse
import cv2
import zipfile
import itertools
import json
import numpy as np


sys.path.insert(1, '../utils/')
from coord_helper import * 
from data_helper import * 
import obj_file

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="../data")
	args = parser.parse_args()

	data_dir = os.path.join(args.home_dir_data, 'geo_data')
	labels_folder_dir = os.path.join(args.home_dir_data, 'geo_data/labels/')
	exclude_dir = os.path.join(args.home_dir_data, 'exclude')
	output_dir = os.path.join(args.home_dir_data, 'collection_result')
	collection_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result')
	visualize_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result_visualize')
	chunk_folder_dir = os.path.join(args.home_dir_data, 'geo_data/misc_chunks')
	labeled_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result_labeled')

	all_hook_name, all_hook_urdf, all_object_name, all_object_urdf = load_all_hooks_objects(data_dir, exclude_dir, labels_folder_dir, True)

	ct = 0

	# for i, hook_name in enumerate(all_hook_name):
	all_commands = []
	# for visualize_labeled_folder_name in os.listdir(labeled_result_folder_dir):
	for hook_name in all_hook_name:
		# for i in range(5):
			# hook_name = visualize_labeled_folder_name.replace('visualize_chunk_', '')
			command = 'python pad_partial_pc_stage1.py --home_dir_data {} --hook_name {}'.format(args.home_dir_data, hook_name)
			all_commands.append(command)

	
	output_command_dir = 'pad_partial_pc_stage1_commands.txt'
	with open(output_command_dir, 'w+') as f:
		for line in all_commands:
			f.write(line + '\n')