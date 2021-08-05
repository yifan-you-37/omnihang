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
	args = parser.parse_args()

	data_dir = os.path.join(args.home_dir_data, 'geo_data')
	collection_result_dir = os.path.join(args.home_dir_data, 'collection_result_vary_scale')
	collection_labels_dir = os.path.join(collection_result_dir, 'labels')

	all_dict = {}

	ct = 0
	for result_file in os.listdir(collection_result_dir):
		if not result_file.endswith('.txt'):
			continue
		result_file_name = result_file[:-4]
		result_np = load_result_file(os.path.join(collection_result_dir, result_file))
		if result_np.size == 0:
			continue
		all_dict[result_file_name] = []

		n_v = 0
		one_v_ct = 0
		prev_scale = None
		cur_v_arr = []
		for i in range(result_np.shape[0]):
			if i == 0:
				one_v_ct += 1
				prev_scale = result_np[i][:6]
				cur_v_arr.append(i)
				continue
			if np.allclose(result_np[i][:6], prev_scale):
				one_v_ct += 1
				cur_v_arr.append(i)
				continue

			all_dict[result_file_name].append(np.array(cur_v_arr))
			cur_v_arr = [i]
			prev_scale = result_np[i][:6] 
			one_v_ct = 1
			n_v += 1
		
		if len(cur_v_arr) > 0:
			n_v += 1
			all_dict[result_file_name].append(np.array(cur_v_arr))
		# print(os.path.join(collection_result_dir, result_file))
		# print(all_dict[result_file_name])
		ct += 1
		if ct % 100 == 0:
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
			for j in range(one_arr[i].shape[0]):
				out_arr.append('{}_v{}_{}'.format(result_file_name, i+1, one_arr[i][j]))

	with open(out_dir, 'w+') as f:
		for line in out_arr:
			f.write(line + '\n')

	print('total v', len(all_result_file_name))
	print('avg', len(out_arr) / len(all_result_file_name))
