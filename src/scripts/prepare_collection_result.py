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
	parser.add_argument("--home_dir_data", default="/home/yifanyou/hang")
	parser.add_argument("--collection_result_dir", default="collection_result_more")
	args = parser.parse_args()

	data_dir = os.path.join(args.home_dir_data, 'geo_data')
	collection_result_dir = os.path.join(args.home_dir_data, args.collection_result_dir)
	collection_labels_dir = os.path.join(collection_result_dir, 'labels')

	print('collection_result_dir', collection_result_dir)
	
	filter_dict_dir = os.path.join(args.home_dir_data, 'collection_result_more_seq', 'labels', 'all_dict.json')
	filter_dict = load_json(filter_dict_dir)

	all_dict = {}

	ct = 0
	out_ct = 0
	for result_file in os.listdir(collection_result_dir):
		if not result_file.endswith('.txt'):
			continue
		result_file_name = result_file[:-4]
		result_np = load_result_file(os.path.join(collection_result_dir, result_file))
		if result_np.size == 0:
			continue
		all_dict[result_file_name] = []

		for ii in range(result_np.shape[0]):
			result_file_name_w_pose = result_file_name + '_' + str(ii)
			# if not (result_file_name_w_pose in filter_dict):
				# continue
			# if not filter_dict[result_file_name_w_pose]:
				# continue
			all_dict[result_file_name].append(ii) 
			out_ct += 1

		# print(all_dict[result_file_name])
		ct += 1
		if ct % 100 == 0:
			print(os.path.join(collection_result_dir, result_file))
			print(ct)
		# if ct >= 5:
		# 	break
	
	out_dir = os.path.join(collection_labels_dir, 'all_list.txt')
	all_result_file_name = list(all_dict.keys())
	random.shuffle(all_result_file_name)
	
	out_arr = []
	for result_file_name in all_result_file_name:
		one_arr = all_dict[result_file_name]
		for i in range(len(one_arr)):
			out_arr.append('{}_{}'.format(result_file_name, one_arr[i]))

	with open(out_dir, 'w+') as f:
		for line in out_arr:
			f.write(line + '\n')

	print('total result file name', len(all_result_file_name))
	print('avg', len(out_arr) / len(all_result_file_name))
