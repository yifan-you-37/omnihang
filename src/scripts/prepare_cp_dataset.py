
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
	args = parser.parse_args()

	home_dir_data = args.home_dir_data
	cp_result_folder_dir = os.path.join(home_dir_data, 'dataset_cp')
	cp_mat_folder_dir = os.path.join(home_dir_data, 'dataset_cp_mat')
	pose_result_folder_dir = os.path.join(home_dir_data, 'collection_result')
	seq_result_folder_dir = os.path.join(args.home_dir_data, 'collection_result_seq')

	out_dir = os.path.join(home_dir_data, 'result_name_n_pose_dict.csv')
	all_data = {}
	ct = 0
	for cp_result_file in os.listdir(cp_result_folder_dir):
		result_file_dir = os.path.join(cp_result_folder_dir, cp_result_file)
		if os.path.getsize(result_file_dir) == 0:
			continue
		
		result_file_name = cp_result_file[:-5]
		# pose_result_file = '{}.txt'.format(result_file_name)
		# pose_result_file_dir = os.path.join(pose_result_folder_dir, pose_result_file)
		# assert os.path.isfile(pose_result_file_dir)
		
		#import time
		#a = time.time()
		try:
			with open(result_file_dir) as f:
				pose_idx = [tmp['index'] for tmp in json.load(f)]
		except:
			continue
		#b = time.time()
		final_pose_idx = []
		for i, k in enumerate(pose_idx):
			# result_json_dir = os.path.join(seq_result_folder_dir, result_file_name + '-' + str(k) + '.json')
			np_dir = os.path.join(seq_result_folder_dir, result_file_name + '-' + str(k) + '-0-pose.npy')
			if os.path.exists(np_dir):
				final_pose_idx.append(i)
				#print(i, pose_idx)
				#assert i in pose_idx
		#print(time.time() - b, b-a)


		all_data[result_file_name] = final_pose_idx
		ct += 1
		print(result_file_dir, final_pose_idx, ct)
		if ct % 500 == 0:
			# break
			print('writing...', len(all_data))
			w = csv.writer(open(out_dir, "w+"))
			for key, val in all_data.items():
				w.writerow([key, *val])
		
	print('writing...', len(all_data))
	w = csv.writer(open(out_dir, "w+"))
	for key, val in all_data.items():
		w.writerow([key, *val])

	
