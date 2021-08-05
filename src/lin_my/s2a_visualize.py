import numpy as np
import sys
import random
import os
import time
import argparse
import glob
import matplotlib.pyplot as plt

try:
	from mayavi import mlab as mayalab 
except:
	pass
np.random.seed(2)

# from contact_point_dataset_torch_multi_label import MyDataset 
from cp_dataset_multi_label_multi_scale import MyDataset 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'utils'))
sys.path.append(UTILS_DIR)
from data_helper import * 
from coord_helper import *
from rotation_lib import *
from bullet_helper import *

import pybullet as p

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--home_dir_data", default="/home/yifanyou/hang")
	parser.add_argument("--exp_name", default="")
	parser.add_argument("--eval_epoch", type=int, default=-1)
	parser.add_argument("--eval_ct", type=int, default=-1)
	parser.add_argument('--test_list', default='test_list')

	parser.add_argument('--restrict_object_cat', default='')

	args = parser.parse_args()

	assert (args.eval_ct != -1) or (args.eval_epoch != -1)
	data_dir = os.path.join(args.home_dir_data, 'geo_data')
	hook_dict, object_dict = load_all_hooks_objects(data_dir, ret_dict=True)

	runs_dir = 'runs/exp_s2a'

	p_env = p_Env(args.home_dir_data, gui=True, physics=False)
	
	for i, run_folder_dir in enumerate(glob.glob('{}/*{}'.format(runs_dir, args.exp_name))):
		# assert i == 0, run_folder_dir
		result_folder = run_folder_dir

		if args.eval_ct != -1:
			eval_file_dir_arr = glob.glob('{}/eval/*_ct_{}.json'.format(run_folder_dir, args.eval_ct))
		elif args.eval_epoch != -1:
			eval_file_dir_arr = glob.glob('{}/eval/*eval_epoch_{}_ct_*.json'.format(run_folder_dir, args.eval_epoch))
		assert len(eval_file_dir_arr) == 1, eval_file_dir_arr
		eval_file_dir = eval_file_dir_arr[0]

		eval_result_dict = load_json(eval_file_dir)

		for result_file_name in eval_result_dict:
			one_result = eval_result_dict[result_file_name]
			# if not 'wrench' in result_file_name:
				# continue
			pc_o = np.array(one_result['pc_o'])
			pc_h = np.array(one_result['pc_h'])
			gt_cp_score_o = np.array(one_result['gt_cp_score_o'])
			gt_cp_score_h = np.array(one_result['gt_cp_score_h'])
			pred_cp_score_o = np.array(one_result['pred_cp_score_o'])
			pred_cp_score_h = np.array(one_result['pred_cp_score_h'])

			pred_cp_top_k_idx_o = pred_cp_score_o.argsort()[-128:][::-1]
			pred_cp_top_k_idx_h = pred_cp_score_h.argsort()[-128:][::-1]

			gt_pose = np.array(one_result['gt_pose'])
			pred_transl = np.array(one_result['pred_transl'])
			pred_aa = np.array(one_result['pred_aa'])

			p_env.load_pair_w_pose(result_file_name, gt_pose[:3], gt_pose[3:], aa=True)
			input('rw')
			p_env.load_pair_w_pose(result_file_name, pred_transl, pred_aa, aa=True)
			print(np.min(pred_cp_score_o), np.max(pred_cp_score_o))
			print(np.min(pred_cp_score_h), np.max(pred_cp_score_h))
			print('mean', np.mean(pred_cp_score_o), np.mean(pred_cp_score_h))
			print('loss o', one_result['loss_o'], 'loss h', one_result['loss_h'])

			# pred_cp_score_o = pred_cp_score_o / np.max(pred_cp_score_o)
			# pred_cp_score_h = pred_cp_score_h / np.max(pred_cp_score_h)
			print('calc o', np.mean(np.abs(pred_cp_score_o - gt_cp_score_o)**2), np.mean(np.abs(pred_cp_score_o/ np.max(pred_cp_score_o) - gt_cp_score_o)**2))
			print('calc o (const)', np.mean(np.abs(0.1 - gt_cp_score_o)**2))
			print('calc h', np.mean(np.abs(pred_cp_score_h - gt_cp_score_h)**2), np.mean(np.abs(pred_cp_score_h/ np.max(pred_cp_score_h) - gt_cp_score_h)**2))
			print('calc h (const)', np.mean(np.abs(0.1 - gt_cp_score_h)**2))
			plot_pc_s(pc_o, gt_cp_score_o)
			mayalab.show()
			plot_pc_s(pc_o, pred_cp_score_o)
			mayalab.show()
			plot_pc_s(pc_o, pred_cp_score_o, abs=False)
			mayalab.show()
			plot_pc(pc_o)
			plot_pc(pc_o[pred_cp_top_k_idx_o], color=[1, 0, 0])
			mayalab.show()


			plot_pc_s(pc_h, gt_cp_score_h)
			mayalab.show()
			plot_pc_s(pc_h, pred_cp_score_h)
			mayalab.show()
			plot_pc_s(pc_h, pred_cp_score_h, abs=False)
			mayalab.show()
			plot_pc(pc_h)
			plot_pc(pc_h[pred_cp_top_k_idx_h], color=[1, 0, 0])
			mayalab.show()
