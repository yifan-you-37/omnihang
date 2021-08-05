import numpy as np
import sys
import random
import os
import time
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'utils'))
sys.path.append(UTILS_DIR)
from data_helper import * 
from train_helper import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--eval_folder_dir")
	args = parser.parse_args()
	
	result_file_name_arr = []
	loss_tracker = LossTrackerMore()
	ct = 0

	loss_dict_all = {}
	for tmp in os.listdir(args.eval_folder_dir):
		if not tmp.endswith('.json'):
			continue
		if not tmp.startswith('hook_'):
			continue
		result_file_name = tmp[:-5]

		ct += 1
		result_dict = load_json(os.path.join(args.eval_folder_dir, tmp))
		# acc_arr.append(result_dict['bullet_flag'])
		loss_tracker.add_dict(result_dict, result_file_name)
		
		loss_dict_all.update({
			result_file_name: result_dict
		})
		# print(tmp)
	print('n processed', ct)
	save_json(os.path.join(args.eval_folder_dir, 'all_eval.json'), loss_dict_all)
	
	loss_tracker.print()