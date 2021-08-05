import time
import numpy as np
import random
# np.random.seed(5)
# random.seed(5)
import sys
import os
import argparse
sys.path.insert(1, '../utils/')
from coord_helper import * 
from data_helper import * 
import bullet_client as bc


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="../data")
	parser.add_argument("--pose_folder_name", default='collection_result')
	args = parser.parse_args()

	data_dir = os.path.join(args.home_dir_data, 'geo_data')
	collection_result_dir = os.path.join(args.home_dir_data, args.pose_folder_name)
	collection_labels_dir = os.path.join(collection_result_dir, 'labels')

	all_dict = {}

	ct = 0
	succ_ct = 0
	fail_ct = 0
	n_pair = 0
	for i, result_file in enumerate(os.listdir(collection_labels_dir)):
		if not result_file.endswith('.json'):
			continue
		if result_file.endswith('all_dict.json'):
			continue
		tmp_dict = load_json(os.path.join(collection_labels_dir, result_file))
		if len(tmp_dict) == 0:
			continue
		n_pair += 1
		for result_file_name in tmp_dict:
			tmp_result = tmp_dict[result_file_name]
			for pose_id in tmp_result:
				result_file_name_w_pose = result_file_name + '_' + str(pose_id)
				if tmp_result[str(pose_id)]:
					all_dict[result_file_name_w_pose] = True
					succ_ct += 1
				else:
					all_dict[result_file_name_w_pose] = False
					# print(result_file_name_w_pose)
					fail_ct += 1
		if i % 100 == 0:
			print(n_pair, succ_ct, fail_ct)
	print(n_pair, succ_ct, succ_ct / n_pair, fail_ct, fail_ct / n_pair)
	
	out_dir = os.path.join(collection_labels_dir, 'all_dict.json')
	save_json(out_dir, all_dict)
	
	
	